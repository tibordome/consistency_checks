#!/bin/bash

sim_list="BL7e22Gas128b40 BL7e22Gas256b40"
snap_list="00 14"
tot_list="${sim_list} ${snap_list}"

echo "Generating and plotting power spectra with nbodykit..."
mpirun -n 10 --mca mpi_warn_on_fork 0 --mca btl_openib_want_fork_support 1 --mca btl openib,self,vader --mca btl_openib_allow_ib 1 python3 plot_pspectra_polySim.py -l "${tot_list}"
echo "Finished generating and plotting power spectra for all sims and snaps requested."