#!/bin/bash

for i in $(seq 0 2); do  # end inclusive
for j in $(seq 0 4); do
    idl <<< "gettf_nbs, /fixed_cfdt, part_nbs_i=$i, part_x_i=$j" &
done
done

wait

for i in $(seq 3 4); do  # end inclusive
for j in $(seq 0 4); do
    idl <<< "gettf_nbs, /fixed_cfdt, part_nbs_i=$i, part_x_i=$j" &
done
done

wait

echo "tqdms end"

echo "part_nbs complete."