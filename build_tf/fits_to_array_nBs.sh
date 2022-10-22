#!/bin/bash

for i in $(seq 0 4); do # end inclusive
for j in $(seq 0 4); do
    python fits_to_array_nBs_A.py $i $j &
done
done
wait

echo "tqdms end"

echo "A: complete!"