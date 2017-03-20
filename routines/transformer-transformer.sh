#!/bin/bash
for a in `ls -1 -S -r matrices/*.mtx`; do
  echo "bin/transformer -i" $a "-o representations/"$(basename $a) "-csr -ellpack";
  
  START=$(date +%s.%N)
  bin/transformer -i $a -o representations/$(basename $a) -csr -ellpack;
  END=$(date +%s.%N)
  
  DIFF=$(echo "$END - $START" | bc)
  echo "Transformed in:" $DIFF
done
