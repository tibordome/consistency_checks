#!/bin/bash

sim_list="BL1e22Gas256b40 BL7e22Gas512b40 BL2e21Gas512b40"
python3 setup.py build_ext --inplace

echo "Plot projections..."
python3 plot_projections.py -l "${sim_list}"
echo "Finished plotting projections, DM (+ gas if applicable), from snap 0 of all simulations requested."