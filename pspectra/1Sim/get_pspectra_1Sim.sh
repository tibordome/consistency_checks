#!/bin/bash

sim="BL7e22Gas256b40 BL7e22Gas256b10"
snap_list=()
for snap in 0{0..9} {10..34} ; do
    snap_list+=("${snap}")
done
snap_list="${snap_list[@]}"
tot_list="${sim} ${snap_list}"

echo "Generating and plotting power spectra with nbodykit..."
mpirun -n 10 --mca mpi_warn_on_fork 0 --mca btl_openib_want_fork_support 1 --mca btl openib,self,vader --mca btl_openib_allow_ib 1 python3 plot_pspectra_1Sim.py -l "${tot_list}"
echo "Finished generating and plotting power spectra for sim(s) ${sim} and all snaps requested."
