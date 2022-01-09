#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 14:42:39 2021

@author: tibor
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 13})
import argparse
import h5py
import os
import time
start_time = time.time()
import re
colormap = plt.cm.gist_ncar
import nbodykit.lab as nbk
from print_msg import print_status
from nbodykit.source.catalog import HDFCatalog
from nbodykit import CurrentMPIComm
comm = CurrentMPIComm.get()
rank = comm.rank
size = comm.size

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
colors = [colormap(i) for i in np.linspace(0.0, 0.9, len(sim_list))]

def plotPSpectra(INPUT_PATH, OUTPUT_DEST, sim_list, snap_list, start_time, gas = False):
    """Plots the pspectra
    Arguments:
    -------------
    grid: (N, N, N)-density array, either calculated via CIC or SPH or DTFE
    INPUT_PATH, OUTPUT_DEST: self-explanatory
    sim_list: list of strings, name of all simulations
    snap_list: list of strings, list of all snaps
    gas: boolean whether we are plotting gas or DM projected densities
    Returns:
    -------------
    1 plot containing all DM / gas spectra"""
        
    if gas == False:
        suffix = 'DM'
        part_type = '1'
    else:
        suffix = 'B'
        part_type = '0'
    
    if rank == 0:
        plt.figure()
    for ind_snap, snap_str in enumerate(snap_list):
        
        print_status(rank,start_time,'Checking whether snapshot {0} exists for all sims..'.format(snap_str))
        exists = True
        for simCase in sim_list:
            if not os.path.isdir('{0}/{1}/snapdir_0{2}'.format(INPUT_PATH, simCase, snap_str)):
                print_status(rank,start_time,'Snapshot {0} does not exist. Continue.'.format(snap_str))
                exists = False
                break
        if exists == False:
            continue
        
        f = h5py.File('{0}/{1}/snapdir_0{2}/snap_0{3}.{4}.hdf5'.format(INPUT_PATH, simCase, snap_str, snap_str, 0), 'r')
        z = f['Header'].attrs['Redshift']
        print_status(rank,start_time,'We are dealing with redshift {0}'.format(z))
        
        
        for ind_sim, simCase in enumerate(sim_list):
            print_status(rank,start_time,'We are dealing with sim {0}'.format(simCase))
            # Figuring out whether this is a gas run, what is N, L_BOX
            res = re.split('(\d+)', simCase)
            N = int(res[-4])
            L_BOX = float(res[-2])
            delx = L_BOX/N
            
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
            
            # Set binning parameters
            k_ny = 2*np.pi/delx/2 # Nyquist frequency
            k_f = 2*np.pi/L_BOX # Fundamenta frequency of the box
            dk = (k_ny - k_f)/300 # Feel free to change 300
    
            # Measure power spectrum using nbodykit's built-in routines
            print_status(rank,start_time,'Call FFTPower')
            r = nbk.FFTPower(in_mesh, mode='1d', kmin = k_f, dk = dk)
            Pk = r.power
            print_status(rank,start_time,'Succeeded.')
    
            if rank == 0:
                if ind_snap == len(snap_list)-1: # Overplotting same curves twice is fine iff alpha = 1.
                    plt.loglog(Pk['k'], Pk['power'].real, color = colors[ind_sim], alpha = 1/(len(snap_list))*(ind_snap+1), label="{0}, {1}".format(simCase, suffix))
                if ind_sim == 0 and (ind_snap == 0 or ind_snap == 1 or ind_snap == len(snap_list) -1):
                    plt.loglog(Pk['k'], Pk['power'].real, color = colors[ind_sim], alpha = 1/(len(snap_list))*(ind_snap+1), label="z={:.2f}".format(z))
                if ind_snap != len(snap_list)-1 and ind_sim != 0:
                    plt.loglog(Pk['k'], Pk['power'].real, color = colors[ind_sim], alpha = 1/(len(snap_list))*(ind_snap+1))
                if ind_snap == 0:
                    plt.axvline(x=k_ny, color = colors[ind_sim], alpha = 0.5) # In case multiple k_ny overlap, alpha = 0.5 helps.
                    plt.axvline(x=k_f, color = colors[ind_sim], linestyle='--', alpha = 0.5)

    if rank == 0:
        plt.legend(fontsize="xx-small")
        plt.ylabel(r"$P(k)$ [$h^{-3}\mathrm{Mpc}^3$]")
        plt.xlabel(r"$k$ [$h \ \mathrm{Mpc}^{-1}$]")
        label_ = ""
        for entry in my_list:
            label_ += entry
            label_ += "_"
        plt.savefig("{0}/{1}pspectra{2}.pdf".format(OUTPUT_DEST, label_, suffix), bbox_inches='tight')

OUTPUT_DEST = '/home/td448/Projects/consistency_checks/get_pspectra/polysim/out'
INPUT_PATH = '/data/highz4/AFdata4/AF_WDM_LS_2021'
res = re.split('(\d+)', sim_list[0]) # First list entry decides whether gas will be investigated too. So better all be "NoGas" or "Gas".
if res[4] == "NoGas":
    gas = False
else:
    assert res[4] == "Gas"
    gas = True
print_status(rank, start_time, 'Get DM power spectra')
plotPSpectra(INPUT_PATH, OUTPUT_DEST, sim_list, snap_list, start_time, gas = False)
if gas == True:
    print_status(rank, start_time, 'Get gas power spectra')
    plotPSpectra(INPUT_PATH, OUTPUT_DEST, sim_list, snap_list, start_time, gas = True)