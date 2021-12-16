#!/bin/bash

sim_loc="/home/td448/rds/rds-dirac-dp140/WDM_pmocz/output"
this_file_loc="/home/td448/rds/rds-dirac-dp140/WDM_pmocz/consistency_checks"
sim_list="BLGas2e21256b40 BLGas7e221024b40 BLGas7e22256b40 BLGas2e21512b40 BLGas7e221024b10"

arr_snap_star=()
for sim in $sim_list; do
    echo "We are now focusing on simulation ${sim}."
    cd "${sim_loc}/${sim}"
    for snap in 0{0..9} {10..50}; do
        if ! [[ -d "snapdir_0${snap}" ]]; then
            break
        fi
        last_snap="$snap"
    done
    echo "The last snap for this sim is ${last_snap}."
    found_stars=false
    for snap in 0{0..9} {10..34}; do
        if [ "$found_stars" = false ] && [ "$snap" -le "$last_snap" ]
        then
            h5dump -n "snapdir_0"$snap"/snap_0"$snap".0.hdf5" > hdf5_dump.txt
            if grep -Fq "PartType4" hdf5_dump.txt
            then
                found_stars=true
                add_snap_star="$snap"
                arr_snap_star+=("$add_snap_star")
                echo "The first snap where stars are present for this sim is ${snap}."
                break
            fi
        fi
    done
    if [ "$found_stars" = false ]; then
        echo "Simulation ${sim} has no stars at all."
    fi
done
echo "We have found the following snap maxes: ${arr_snap_star[@]}"
max=${arr_snap_star[0]}
for n in "${arr_snap_star[@]}" ; do
    [[ $n > $max ]] && max=$n
done
echo "The maximum of these numbers is $max."
cd "$this_file_loc"


