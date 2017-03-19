#!/bin/bash
for a in `ls -1 matrices/*.mtx`; do
  echo "bin/transformer -i" $a "-o representations/"$(basename $a) "-csr -ellpack";
  bin/transformer -i $a -o representations/$(basename $a) -csr -ellpack;
done
