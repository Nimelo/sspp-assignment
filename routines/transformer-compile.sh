#!/bin/bash
echo "source /apps/intel/xe2017/config_intel.sh"
source /apps/intel/xe2017/config_intel.sh
echo "icpc -std=c++11 -o transformer code/transformer/*.cpp code/common/*cpp code/common/*h"
icpc -std=c++11 -o bin/transformer code/transformer/*.cpp code/common/*cpp code/common/*h