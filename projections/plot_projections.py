#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 09:58:44 2021

@author: tibor
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 13})
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import h5py
import re
import argparse
import make_grid_cic

parser = argparse.ArgumentParser()
parser.add_argument('-l', '--list', help='delimited list input', type=str)
args = parser.parse_args()
my_list = [item for item in args.list.split(' ')]

def getHDF5Data(path, with_gas = True):
        
    dm_x = np.empty(0, dtype = np.float32)
    dm_y = np.empty(0, dtype = np.float32)
    dm_z = np.empty(0, dtype = np.float32)
    if with_gas == True:
        gas_masses = np.empty(0, dtype = np.float32)
        gas_x = np.empty(0, dtype = np.float32)
        gas_y = np.empty(0, dtype = np.float32)
        gas_z = np.empty(0, dtype = np.float32)
    for snap_run in range(16):
        f = h5py.File('{0}/snap_000.{1}.hdf5'.format(path, snap_run), 'r')
        dm_x = np.hstack((dm_x, np.float32(f['PartType1/Coordinates'][:,0]/1000))) # in Mpc = 3.085678e+27 cm
        dm_y = np.hstack((dm_y, np.float32(f['PartType1/Coordinates'][:,1]/1000))) 
        dm_z = np.hstack((dm_z, np.float32(f['PartType1/Coordinates'][:,2]/1000))) 
        if with_gas == True:
            gas_masses = np.hstack((gas_masses, np.float32(f['PartType0/Masses'][:]))) # in Mpc = 3.085678e+27 cm
            gas_x = np.hstack((gas_x, np.float32(f['PartType0/Coordinates'][:,0]/1000))) # in Mpc = 3.085678e+27 cm
            gas_y = np.hstack((gas_y, np.float32(f['PartType0/Coordinates'][:,1]/1000))) 
            gas_z = np.hstack((gas_z, np.float32(f['PartType0/Coordinates'][:,2]/1000))) 
        
    dm_xyz = np.hstack((np.reshape(dm_x, (dm_x.shape[0],1)), np.reshape(dm_y, (dm_y.shape[0],1)), np.reshape(dm_z, (dm_z.shape[0],1))))
    gas_xyz = np.hstack((np.reshape(gas_x, (gas_x.shape[0],1)), np.reshape(gas_y, (gas_y.shape[0],1)), np.reshape(gas_z, (gas_z.shape[0],1))))
    dm_masses = np.ones((dm_xyz.shape[0],), dtype=np.float32)*np.float32(f['Header'].attrs['MassTable'][1]) # in 1.989e+43 g
    
    if with_gas == True:
        return dm_xyz, dm_masses, gas_xyz, gas_masses
    else:
        return dm_xyz, dm_masses, None, None

def plotGridProjections(grid, N, L_BOX, OUTPUT_DEST, sim, gas):
    """Plots the grid projections
    Arguments:
    -------------
    grid: (N, N, N)-density array, either calculated via CIC or SPH or DTFE
    N, L_BOX, OUTPUT_DEST: self-explanatory
    sim: string, name of simulation
    gas: boolean whether we are plotting gas or DM projected densities
    Returns: 
    -------------
    3 grid projection plots"""
    
    if gas == True:
        suffix = 'B'
    else:
        suffix = 'DM'
    rho_proj_cic = np.zeros((N, N))
    for h in range(N):
        rho_proj_cic += grid[h,:,:]
    rho_proj_cic /= N
    plt.figure()
    plt.imshow(rho_proj_cic,interpolation='None',origin='upper', extent=[0, L_BOX, L_BOX, 0], cmap="hot")
    plt.gca().xaxis.set_major_locator(MultipleLocator(L_BOX/4))
    plt.gca().xaxis.set_minor_locator(MultipleLocator(L_BOX/20))
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f')) # float
    plt.gca().yaxis.set_major_locator(MultipleLocator(L_BOX/4))
    plt.gca().yaxis.set_minor_locator(MultipleLocator(L_BOX/20))
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f')) # float
    plt.xlabel(r"z (cMpc/h)", fontweight='bold')
    plt.ylabel(r"y (cMpc/h)", fontweight='bold')
    plt.colorbar()
    plt.title(r'{0}, x-Projected {1}-Density'.format(sim, suffix))
    plt.savefig("{0}/{1}xProjRho{2}.pdf".format(OUTPUT_DEST, sim, suffix))
    
    rho_proj_cic = np.zeros((N, N))
    for h in range(N):
        rho_proj_cic += grid[:,h,:]
    rho_proj_cic /= N
    plt.figure()
    plt.imshow(rho_proj_cic,interpolation='None',origin='upper', extent=[0, L_BOX, L_BOX, 0], cmap="hot")
    plt.gca().xaxis.set_major_locator(MultipleLocator(L_BOX/4))
    plt.gca().xaxis.set_minor_locator(MultipleLocator(L_BOX/20))
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f')) # float
    plt.gca().yaxis.set_major_locator(MultipleLocator(L_BOX/4))
    plt.gca().yaxis.set_minor_locator(MultipleLocator(L_BOX/20))
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f')) # float
    plt.xlabel(r"z (cMpc/h)", fontweight='bold')
    plt.ylabel(r"x (cMpc/h)", fontweight='bold')
    plt.colorbar()
    plt.title(r'{0}, y-Projected {1}-Density'.format(sim, suffix))
    plt.savefig("{0}/{1}yProjRho{2}.pdf".format(OUTPUT_DEST, sim, suffix))
    
    rho_proj_cic = np.zeros((N, N))
    for h in range(N):
        rho_proj_cic += grid[:,:,h]
    rho_proj_cic /= N
    plt.figure()
    plt.imshow(rho_proj_cic,interpolation='None',origin='upper', extent=[0, L_BOX, L_BOX, 0], cmap="hot")
    plt.gca().xaxis.set_major_locator(MultipleLocator(L_BOX/4))
    plt.gca().xaxis.set_minor_locator(MultipleLocator(L_BOX/20))
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f')) # float
    plt.gca().yaxis.set_major_locator(MultipleLocator(L_BOX/4))
    plt.gca().yaxis.set_minor_locator(MultipleLocator(L_BOX/20))
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f')) # float
    plt.xlabel(r"y (cMpc/h)", fontweight='bold')
    plt.ylabel(r"x (cMpc/h)", fontweight='bold')
    plt.colorbar()
    plt.title(r'{0}, z-Projected {1}-Density'.format(sim, suffix))
    plt.savefig("{0}/{1}zProjRho{2}.pdf".format(OUTPUT_DEST, sim, suffix))


for simCase in my_list:
    print('We are dealing with sim {0}'.format(simCase))
    # Figuring out whether this is a gas run, what is N, L_BOX
    res = re.split('(\d+)', simCase)
    if res[4] == "NoGas":
        gas_run = False
    else:
        assert res[4] == "Gas"
        gas_run = True
    N = int(res[-4])
    L_BOX = float(res[-2])
    dm_xyz, dm_masses, gas_xyz, gas_masses = getHDF5Data('/home/td448/rds/rds-dirac-dp140/WDM_pmocz/output/{0}/snapdir_000'.format(simCase), with_gas = gas_run)
    OUTPUT_DEST = '/home/td448/rds/rds-dirac-dp140/WDM_pmocz/consistency_checks/get_proj_rho/out'
    # DM
    grid = make_grid_cic.makeGridWithCICPBC(dm_xyz[:,0].astype('float32'), dm_xyz[:,1].astype('float32'), dm_xyz[:,2].astype('float32'), dm_masses.astype('float32'), L_BOX, N)
    plotGridProjections(grid, N, L_BOX, OUTPUT_DEST, simCase, gas = False)
    # Baryons
    if gas_run == True:
        grid = make_grid_cic.makeGridWithCICPBC(gas_xyz[:,0].astype('float32'), gas_xyz[:,1].astype('float32'), gas_xyz[:,2].astype('float32'), gas_masses.astype('float32'), L_BOX, N)
        plotGridProjections(grid, N, L_BOX, OUTPUT_DEST, simCase, gas = True)