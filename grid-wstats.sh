#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 <runseed>"
    exit 1
fi

p_values=("01" "03" "05" "07" "1")
r_values=("05" "1" "2" "3" "4")

for p in "${p_values[@]}"
do
    for r in "${r_values[@]}"
    do
        python sweep-wstats-2.py "lif_alpha_beta_1_different_net_seed_0_pinh_${p}_rinh_${r}_runseed_$1" 300 15
    done
done