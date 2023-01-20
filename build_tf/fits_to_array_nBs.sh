#!/bin/bash

for i in $(seq 0 3); do # end inclusive
for j in $(seq 0 9); do
    python fits_to_array_nBs_A.py $i $j &
done
done
wait

for i in $(seq 4 6); do # end inclusive
for j in $(seq 0 9); do
    python fits_to_array_nBs_A.py $i $j &
done
done
wait

for i in $(seq 7 9); do # end inclusive
for j in $(seq 0 9); do
    python fits_to_array_nBs_A.py $i $j &
done
done
wait

echo "tqdms end"

echo "A: complete!"