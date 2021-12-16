#!/bin/bash

sim_list="BL7e22Gas128b40 BL7e22Gas256b40"

echo "Generating and plotting power spectra with nbodykit..."
mpirun -n 10 --mca mpi_warn_on_fork 0 --mca btl_openib_want_fork_support 1 --mca btl openib,self,vader --mca btl_openib_allow_ib 1 python3 plot_pspectra_nbodykit.py -l "${sim_list}"
echo "Finished generating and plotting power spectra from snap 0 of all simulations requested."