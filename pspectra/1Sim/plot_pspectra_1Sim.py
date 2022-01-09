#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 14:42:39 2021

@author: tibor
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({'font.size': 13})
import argparse
from copy import deepcopy
import h5py
import time
start_time = time.time()
import re
import nbodykit.lab as nbk
from print_msg import print_status
import os
from nbodykit.source.catalog import HDFCatalog
from nbodykit import CurrentMPIComm
comm = CurrentMPIComm.get()
rank = comm.rank
size = comm.size

# User input
OUTPUT_DEST = '/home/td448/Projects/consistency_checks/get_pspectra/1sim/out'
INPUT_PATH = '/data/highz4/AFdata4/AF_WDM_LS_2021'

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('-l', '--list', help='delimited list input', type=str)
args = parser.parse_args()
my_list = [item for item in args.list.split(' ')]
sim_list = []
snap_list = []
for entry in my_list:
    if entry[0] == "L" or entry[0] == "B":
        sim_list.append(entry)
    else:
        snap_list.append(entry)

# Configure discrete colorbar
cmap = plt.get_cmap("jet", len(snap_list))
norm = mpl.colors.BoundaryNorm(np.arange(0,len(snap_list)+1)-0.5, len(snap_list))
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

def plotPSpectra(INPUT_PATH, OUTPUT_DEST, simCase, snap_list, z_list, start_time, gas = False):
    """Plots the pspectra
    Arguments:
    -------------
    grid: (N, N, N)-density array, either calculated via CIC or SPH or DTFE
    INPUT_PATH, OUTPUT_DEST: self-explanatory
    simCase: string, sim name, e.g. BL7e22Gas256b40
    snap_list: list of strings, name of all simulations
    gas: boolean whether we are plotting gas or DM projected densities
    Returns:
    -------------
    1 plot containing all DM / gas spectra"""
    
    z_l = deepcopy(z_list)
    # Figuring out whether this is a gas run, what is N, L_BOX
    res = re.split('(\d+)', simCase)
    if gas == False:
        suffix = 'DM'
        part_type = '1'
    else:
        suffix = 'B'
        part_type = '0'
    N = int(res[-4])
    L_BOX = float(res[-2])
    delx = L_BOX/N
        
    # Set binning parameters
    k_ny = 2*np.pi/delx/2 # Nyquist frequency
    k_f = 2*np.pi/L_BOX # Fundamental frequency of the box
    dk = (k_ny - k_f)/300 # Feel free to change 300
     
    if rank == 0:
        plt.figure()
    for ind, snap_str in enumerate(snap_list):
        print_status(rank,start_time,'Checking whether snapshot {0} exists..'.format(snap_str))
        if not os.path.isdir('{0}/{1}/snapdir_0{2}'.format(INPUT_PATH, simCase, snap_str)):
            print_status(rank,start_time,'Snapshot {0} does not exist. Continue.'.format(snap_str))
            continue
        
        f = h5py.File('{0}/{1}/snapdir_0{2}/snap_0{3}.{4}.hdf5'.format(INPUT_PATH, simCase, snap_str, snap_str, 0), 'r')
        z = f['Header'].attrs['Redshift']
        # Overwrite z_l, but not original z_list
        z_l[ind] = z
        print_status(rank,start_time,'We are dealing with sim {} at redshift {:.2f}'.format(simCase, z))
        
        # Creating Catalog (the catalog module is a bit buggy but we trust that all particles are included given that csize is correct)
        in_cat = HDFCatalog('{0}/{1}/snapdir_0{2}/snap_'.format(INPUT_PATH, simCase, snap_str)+'*', dataset='PartType{0}'.format(part_type))
        print_status(rank, start_time, 'In our nbodykit cat we have {0} {1} particles'.format(in_cat.csize, suffix))
        
        # Convert particle positions from kpc/h to Mpc/h
        in_cat['Coordinates'] /= 1e3

        # Create mesh for density grid
        print_status(rank,start_time,'Painting onto a CIC grid...')
        if gas == False: # DM masses are always the same for CDM, WDM
            in_mesh = in_cat.to_mesh(Nmesh=N, BoxSize=L_BOX, resampler='cic',compensated=True, position='Coordinates')
        else: # Allowing for snap not 0 where baryon masses will start to differ
            in_mesh = in_cat.to_mesh(Nmesh=N, BoxSize=L_BOX, resampler='cic',compensated=True, position='Coordinates', weight='Masses')

        # Measure power spectrum using nbodykit's built-in routines
        print_status(rank,start_time,'Call FFTPower')
        r = nbk.FFTPower(in_mesh, mode='1d', kmin = k_f, dk = dk)
        Pk = r.power
        print_status(rank,start_time,'Succeeded.')
        if rank == 0:
            plt.loglog(Pk['k'], Pk['power'].real, color = cmap(ind))
            plt.axvline(x=k_ny, color = cmap(ind), alpha = 0.5) # In case multiple k_ny overlap, alpha = 0.5 helps.
            plt.axvline(x=k_f, color = cmap(ind), linestyle='--')

            # Saving power spectrum
            np.savetxt("{0}/txt_files/{1}_{2}_psp.txt".format(OUTPUT_DEST, simCase, snap_str), Pk['power'].real, fmt='%1.7e')
            np.savetxt("{0}/txt_files/{1}_{2}_k.txt".format(OUTPUT_DEST, simCase, snap_str), Pk['k'], fmt='%1.7e')
    
    if rank == 0:
        plt.ylabel(r"$P(k)$ [$h^{-3}\mathrm{Mpc}^3$]")
        plt.xlabel(r"$k$ [$h \ \mathrm{Mpc}^{-1}$]")
        plt.title("{0}, {1}".format(simCase, suffix))
        cbar = plt.colorbar(sm, ticks = np.arange(0,len(snap_list)))
        cbar.ax.set_ylabel('z', rotation=270, labelpad=10)
        z_l = ['{:.2f}'.format(z) for z in z_list]
        cbar.set_ticklabels(z_l)
        cbar.ax.tick_params(labelsize='xx-small')
        plt.savefig("{0}/pspectra{1}{2}.pdf".format(OUTPUT_DEST, simCase, suffix), bbox_inches='tight')

# Construct z_list
z_list = []
simCase = 'BL2e21NoGas256b10'
for snap_str in snap_list:
    INPUT_PATH_z = '/data/highz4/AFdata4/AF_WDM_LS_2021'
    f = h5py.File('{0}/{1}/snapdir_0{2}/snap_0{3}.{4}.hdf5'.format(INPUT_PATH_z, simCase, snap_str, snap_str, 0), 'r')
    z = f['Header'].attrs['Redshift']
    z_list.append(z)
np.savetxt("{0}/txt_files/{1}_z_list.txt".format(OUTPUT_DEST, simCase), z_list, fmt='%1.7e')

for simCase in sim_list:
    res = re.split('(\d+)', simCase) # First list entry = sim name.
    if res[0] == 'LLNoGas':
        gas = False
    elif res[0] == 'LLGas':
        gas = True
    else:
        assert res[0] == 'BL'
        if res[4] == "NoGas":
            gas = False
        else:
            assert res[4] == "Gas"
            gas = True
    print_status(rank, start_time, 'Get DM power spectra for all z provided')
    plotPSpectra(INPUT_PATH, OUTPUT_DEST, simCase, snap_list, z_list, start_time, gas = False)
    if gas == True:
        print_status(rank, start_time, 'Get gas power spectra for all z provided')
        plotPSpectra(INPUT_PATH, OUTPUT_DEST, simCase, snap_list, z_list, start_time, gas = True)
