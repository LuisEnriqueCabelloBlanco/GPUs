#!/bin/bash
make

rm db.csv

for i in {10..50}
do
    size=$((i*16))
    value=$(./matrix_mult $size $size $size) 
    echo "$value"
    echo "$value" >> db.csv
done