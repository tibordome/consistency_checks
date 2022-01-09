#!/bin/bash

sim_list="BL1e22NoGas1024b40 BL2e21Gas256b40 BL7e22Gas256b10 BL7e22Gas256b40"

echo "Generating and plotting power spectra with nbodykit..."
mpirun -n 10  python3 plot_pspectra_1z.py -l "${sim_list}"
echo "Finished generating and plotting power spectra from snap 0 of all simulations requested."
