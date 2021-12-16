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
import time
start_time = time.time()
import re
colormap = plt.cm.gist_ncar
import nbodykit.lab as nbk
from print_msg import print_status
from nbodykit.source.catalog import HDFCatalog
from nbodykit import CurrentMPIComm
from mpi4py import MPI #!!! Should be after loading nbodykit, to be on the safe side!!!
comm = CurrentMPIComm.get()
rank = comm.rank
size = comm.size

parser = argparse.ArgumentParser()
parser.add_argument('-l', '--list', help='delimited list input', type=str)
args = parser.parse_args()
my_list = [item for item in args.list.split(' ')]
colors = [colormap(i) for i in np.linspace(0.0, 0.9, len(my_list))]

def getHDF5Data(path, with_gas = True):

    dm_x = np.empty(0, dtype = np.float32)
    dm_y = np.empty(0, dtype = np.float32)
    dm_z = np.empty(0, dtype = np.float32)
    dm_masses = np.empty(0, dtype = np.float32)
    gas_x = np.empty(0, dtype = np.float32)
    gas_y = np.empty(0, dtype = np.float32)
    gas_z = np.empty(0, dtype = np.float32)
    gas_masses = np.empty(0, dtype = np.float32)
    nb_jobs_to_do = 16
    perrank = nb_jobs_to_do//size + (nb_jobs_to_do//size == 0)*1
    do_sth = rank <= nb_jobs_to_do-1
    count = 0
    count_gas = 0
    if size <= nb_jobs_to_do:
        last = rank == size - 1 # Whether or not last process
    else:
        last = rank == nb_jobs_to_do - 1
    for snap_run in range(rank*perrank, rank*perrank+do_sth*(perrank+last*(nb_jobs_to_do-(rank+1)*perrank))):
        f = h5py.File('{0}/snap_000.{1}.hdf5'.format(path, snap_run), 'r')
        dm_x = np.hstack((dm_x, np.float32(f['PartType1/Coordinates'][:,0]/1000))) # in Mpc = 3.085678e+27 cm
        dm_y = np.hstack((dm_y, np.float32(f['PartType1/Coordinates'][:,1]/1000)))
        dm_z = np.hstack((dm_z, np.float32(f['PartType1/Coordinates'][:,2]/1000)))
        dm_masses = np.hstack((dm_masses, np.ones((f['PartType1/Coordinates'][:].shape[0],), dtype=np.float32)*np.float32(f['Header'].attrs['MassTable'][1]))) # in 1.989e+43 g
        if with_gas == True:
            gas_x = np.hstack((gas_x, np.float32(f['PartType0/Coordinates'][:,0]/1000))) # in Mpc = 3.085678e+27 cm
            gas_y = np.hstack((gas_y, np.float32(f['PartType0/Coordinates'][:,1]/1000)))
            gas_z = np.hstack((gas_z, np.float32(f['PartType0/Coordinates'][:,2]/1000)))
            gas_masses = np.hstack((gas_masses, np.float32(f['PartType0/Masses'][:])))
            count_gas += f['PartType0/Coordinates'][:].shape[0]
        count += f['PartType1/Coordinates'][:].shape[0]

    count_new = comm.gather(count, root=0)
    count_new = comm.bcast(count_new, root = 0)
    nb_dm_ptcs = np.sum(np.array(count_new))
    comm.Barrier()
    recvcounts = np.array(count_new)
    rdispls = np.zeros_like(recvcounts)
    for j in range(rdispls.shape[0]):
        rdispls[j] = np.sum(recvcounts[:j])
    count_new_gas = comm.gather(count_gas, root=0)
    count_new_gas = comm.bcast(count_new_gas, root = 0)
    nb_gas_ptcs = np.sum(np.array(count_new_gas))
    comm.Barrier()
    recvcounts_gas = np.array(count_new_gas)
    rdispls_gas = np.zeros_like(recvcounts_gas)
    for j in range(rdispls_gas.shape[0]):
        rdispls_gas[j] = np.sum(recvcounts_gas[:j])

    dm_x_total = np.empty(nb_dm_ptcs, dtype = np.float32)
    dm_y_total = np.empty(nb_dm_ptcs, dtype = np.float32)
    dm_z_total = np.empty(nb_dm_ptcs, dtype = np.float32)
    dm_masses_total = np.empty(nb_dm_ptcs, dtype = np.float32)
    dm_x_total = np.ascontiguousarray(dm_x_total, dtype = np.float32)
    dm_y_total = np.ascontiguousarray(dm_y_total, dtype = np.float32)
    dm_z_total = np.ascontiguousarray(dm_z_total, dtype = np.float32)
    dm_masses_total = np.ascontiguousarray(dm_masses_total, dtype = np.float32)

    gas_x_total = np.empty(nb_gas_ptcs, dtype = np.float32)
    gas_y_total = np.empty(nb_gas_ptcs, dtype = np.float32)
    gas_z_total = np.empty(nb_gas_ptcs, dtype = np.float32)
    gas_masses_total = np.empty(nb_gas_ptcs, dtype = np.float32)
    gas_x_total = np.ascontiguousarray(gas_x_total, dtype = np.float32)
    gas_y_total = np.ascontiguousarray(gas_y_total, dtype = np.float32)
    gas_z_total = np.ascontiguousarray(gas_z_total, dtype = np.float32)
    gas_masses_total = np.ascontiguousarray(gas_masses_total, dtype = np.float32)

    comm.Gatherv(dm_x, [dm_x_total, recvcounts, rdispls, MPI.FLOAT], root = 0)
    comm.Gatherv(dm_y, [dm_y_total, recvcounts, rdispls, MPI.FLOAT], root = 0)
    comm.Gatherv(dm_z, [dm_z_total, recvcounts, rdispls, MPI.FLOAT], root = 0)
    comm.Gatherv(dm_masses, [dm_masses_total, recvcounts, rdispls, MPI.FLOAT], root = 0)

    comm.Gatherv(gas_x, [gas_x_total, recvcounts_gas, rdispls_gas, MPI.FLOAT], root = 0)
    comm.Gatherv(gas_y, [gas_y_total, recvcounts_gas, rdispls_gas, MPI.FLOAT], root = 0)
    comm.Gatherv(gas_z, [gas_z_total, recvcounts_gas, rdispls_gas, MPI.FLOAT], root = 0)
    comm.Gatherv(gas_masses, [gas_masses_total, recvcounts_gas, rdispls_gas, MPI.FLOAT], root = 0)

    pieces = 1 + (nb_dm_ptcs>=3*10**8)*nb_dm_ptcs//(3*10**8) # Not too high since this is a slow-down!
    pieces_gas = 1 + (nb_gas_ptcs>=3*10**8)*nb_gas_ptcs//(3*10**8) # Not too high since this is a slow-down!
    chunk = nb_dm_ptcs//pieces
    chunk_gas = nb_gas_ptcs//pieces_gas
    dm_x = np.empty(0, dtype = np.float32)
    dm_y = np.empty(0, dtype = np.float32)
    dm_z = np.empty(0, dtype = np.float32)
    dm_masses = np.empty(0, dtype = np.float32)
    gas_x = np.empty(0, dtype = np.float32)
    gas_y = np.empty(0, dtype = np.float32)
    gas_z = np.empty(0, dtype = np.float32)
    gas_masses = np.empty(0, dtype = np.float32)
    for i in range(pieces):
        to_bcast = dm_x_total[i*chunk:(i+1)*chunk+(i==(pieces-1))*(nb_dm_ptcs-pieces*chunk)]
        comm.Bcast(to_bcast, root=0)
        dm_x = np.hstack((dm_x, to_bcast))
        to_bcast = dm_y_total[i*chunk:(i+1)*chunk+(i==(pieces-1))*(nb_dm_ptcs-pieces*chunk)]
        comm.Bcast(to_bcast, root=0)
        dm_y = np.hstack((dm_y, to_bcast))
        to_bcast = dm_z_total[i*chunk:(i+1)*chunk+(i==(pieces-1))*(nb_dm_ptcs-pieces*chunk)]
        comm.Bcast(to_bcast, root=0)
        dm_z = np.hstack((dm_z, to_bcast))
        to_bcast = dm_masses_total[i*chunk:(i+1)*chunk+(i==(pieces-1))*(nb_dm_ptcs-pieces*chunk)]
        comm.Bcast(to_bcast, root=0)
        dm_masses = np.hstack((dm_masses, to_bcast))

    for i in range(pieces_gas):
        to_bcast = gas_x_total[i*chunk_gas:(i+1)*chunk_gas+(i==(pieces_gas-1))*(nb_gas_ptcs-pieces_gas*chunk_gas)]
        comm.Bcast(to_bcast, root=0)
        gas_x = np.hstack((gas_x, to_bcast))
        to_bcast = gas_y_total[i*chunk_gas:(i+1)*chunk_gas+(i==(pieces_gas-1))*(nb_gas_ptcs-pieces_gas*chunk_gas)]
        comm.Bcast(to_bcast, root=0)
        gas_y = np.hstack((gas_y, to_bcast))
        to_bcast = gas_z_total[i*chunk_gas:(i+1)*chunk_gas+(i==(pieces_gas-1))*(nb_gas_ptcs-pieces_gas*chunk_gas)]
        comm.Bcast(to_bcast, root=0)
        gas_z = np.hstack((gas_z, to_bcast))
        to_bcast = gas_masses_total[i*chunk_gas:(i+1)*chunk_gas+(i==(pieces_gas-1))*(nb_gas_ptcs-pieces_gas*chunk_gas)]
        comm.Bcast(to_bcast, root=0)
        gas_masses = np.hstack((gas_masses, to_bcast))

    dm_xyz = np.hstack((np.reshape(dm_x, (dm_x.shape[0],1)), np.reshape(dm_y, (dm_y.shape[0],1)), np.reshape(dm_z, (dm_z.shape[0],1))))
    gas_xyz = np.hstack((np.reshape(gas_x, (gas_x.shape[0],1)), np.reshape(gas_y, (gas_y.shape[0],1)), np.reshape(gas_z, (gas_z.shape[0],1))))

    return dm_xyz, dm_masses, gas_xyz, gas_masses


def plotPSpectra(INPUT_PATH, OUTPUT_DEST, my_list, start_time, gas = False):
    """Plots the pspectra
    Arguments:
    -------------
    grid: (N, N, N)-density array, either calculated via CIC or SPH or DTFE
    INPUT_PATH, OUTPUT_DEST: self-explanatory
    my_list: string, name of all simulations
    gas: boolean whether we are plotting gas or DM projected densities
    Returns:
    -------------
    1 plot containing all DM / gas spectra"""

    if rank == 0:
        plt.figure()
    for ind, simCase in enumerate(my_list):
        print_status(rank,start_time,'We are dealing with sim {0}'.format(simCase))
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
        
        # Creating Catalog (the catalog module is a bit buggy but we trust that all particles are included given that csize is correct)
        in_cat = HDFCatalog('{0}/{1}/snapdir_000/snap_'.format(INPUT_PATH, simCase)+'*', dataset='PartType{0}'.format(part_type))
        print_status(rank, start_time, 'In our nbodykit cat we have columns {0} and {1} {2} particles'.format(in_cat.columns, in_cat.csize, suffix))
        
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
            plt.loglog(Pk['k'], Pk['power'].real, color = colors[ind], label="{0}".format(simCase))
            plt.axvline(x=k_ny, color = colors[ind], alpha = 0.5) # In case multiple k_ny overlap, alpha = 0.5 helps.
            plt.axvline(x=k_f, color = colors[ind], linestyle='--', alpha = 0.5)
            plt.legend(fontsize="small")

    plt.ylabel(r"$P(k)$ [$h^{-3}\mathrm{Mpc}^3$]")
    plt.xlabel(r"$k$ [$h \ \mathrm{Mpc}^{-1}$]")
    plt.savefig("{0}/pspectra{1}.pdf".format(OUTPUT_DEST, suffix), bbox_inches='tight')

OUTPUT_DEST = '/home/td448/Projects/consistency_checks/get_pspectra/out'
INPUT_PATH = '/data/highz4/AFdata4/AF_WDM_LS_2021'
res = re.split('(\d+)', my_list[0]) # First list entry decides whether gas will be investigated too. So better all be "NoGas" or "Gas".
if res[4] == "NoGas":
    gas = False
else:
    assert res[4] == "Gas"
    gas = True
print_status(rank, start_time, 'Get DM power spectra')
plotPSpectra(INPUT_PATH, OUTPUT_DEST, my_list, start_time, gas = False)
if gas == True:
    print_status(rank, start_time, 'Get gas power spectra')
    plotPSpectra(INPUT_PATH, OUTPUT_DEST, my_list, start_time, gas = True)