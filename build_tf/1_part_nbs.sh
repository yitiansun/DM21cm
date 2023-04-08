#!/bin/bash

for i in $(seq 0 14); do  # end inclusive
    idl <<< "gettf_nbs, part_i=$i" &
done

wait

echo "tqdms end"

echo "part_nbs complete."