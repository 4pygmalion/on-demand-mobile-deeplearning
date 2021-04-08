#!/bin/bash

ITER=$(seq 0 10)
runtimes=()

for i in $ITER
do
    START=$(date +%s.%N)
    
    python3 ./profiling_buildgraph.py  
    
    END=$(date +%s.%N)
    runtime=$(echo "$END - $START" | bc)
    echo "$runtime" >> ./result/runtimes_client.txt
    
done