#!/bin/bash

for i in $(seq 0 2); do  # end inclusive
    idl <<< "gettf_nbs, part_i=$i" &
done

wait

echo "tqdms end"

echo "part_nbs complete."