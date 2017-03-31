#!/bin/bash
echo "source /apps/intel/xe2017/config_intel.sh"
source /apps/intel/xe2017/config_intel.sh
START=$(date +%s.%N)
echo "icpc -std=c++11 -O3 -o transformer code/transformer/*.cpp code/common/*cpp code/common/*h"
icpc -std=c++11 -O3 -o bin/transformer code/transformer/*.cpp code/common/*cpp code/common/*h
END=$(date +%s.%N)
DIFF=$(echo "$END - $START" | bc)
echo $DIFF