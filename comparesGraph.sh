#!/bin/bash

K=8
EXEC=./fixedKCoreComputation

graphs=(
  "1000 2995 testGraph_1000_2995.txt"
  "10000 49985 testGraph_10000_49985.txt"
  "100000 499985 testGraph_100000_499985.txt"
  "1000000 4999985 testGraph_1000000_4999985.txt"
  "10000000 49999985 testGraph_10000000_49999985.txt"
)

printf "\n%-10s | %-10s | %-10s | %-10s | %-10s\n" "NODES" "EDGES" "CPU(ms)"  "GPU(ms)" "SPEEDUP" >  DiffgraphResults.txt

printf -- "-------------------------------------------------------------\n" >> DiffgraphResults.txt

for g in "${graphs[@]}"; do
    # g = "nodes edges file"
    $EXEC $g $K > temp.txt

    cpu=$(grep "CPU time" temp.txt | awk '{print $3}')
    gpu=$(grep "GPU time" temp.txt | awk '{print $3}')
    speedup=$(echo "$cpu / $gpu" | bc -l)

    nodes=$(echo $g | awk '{print $1}')
    edges=$(echo $g | awk '{print $2}')

    printf "%-10s | %-10s | %-10s | %-10s | %-10.2f\n" "$nodes" "$edges" "$cpu" "$gpu" "$speedup" >> DiffgraphResults.txt

done

printf -- "-------------------------------------------------------------\n" >> DiffgraphResults.txt

rm temp.txt
