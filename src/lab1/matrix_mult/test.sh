#!/bin/bash
make
for i in {1..10}
do
    size=$((i*16))
    ./matrix_mult $size $size $size
done