#!/bin/bash
for a in `ls -1 *.tar.gz`; do
  echo "tar -zxvf " $a;
  #tar -zxvf $a;
done
for a in `ls -1 *.mtx.gz`; do
  echo "gzip -d " $a;
  gzip -d $a;
done
