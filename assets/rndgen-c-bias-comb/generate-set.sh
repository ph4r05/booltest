#!/bin/bash

BASEDIR="."
SIZE=102400

if [ ! -z "$1" ]; then
    BASEDIR="$1"
fi

echo "Using basedir: $BASEDIR"
echo "Using size: $SIZE"

./rand 1 1 1 1 | dd of=${BASEDIR}/randc_seed1_prob1_x0.data bs=1024 count=$SIZE
./rand 1 1 1 2 | dd of=${BASEDIR}/randc_seed1_prob1_xor_x0_x42.data bs=1024 count=$SIZE
./rand 1 1 1 3 | dd of=${BASEDIR}/randc_seed1_prob1_xor_x0_x42_x100.data bs=1024 count=$SIZE
./rand 1 1 1 4 | dd of=${BASEDIR}/randc_seed1_prob1_xor_x0_x42_x100_x127.data bs=1024 count=$SIZE
./rand 1 1 1 5 | dd of=${BASEDIR}/randc_seed1_prob1_xor_x0_x42_x100_x127_x91.data bs=1024 count=$SIZE

./rand 1 2 1 2 | dd of=${BASEDIR}/randc_seed1_prob1_and_x0_x42.data bs=1024 count=$SIZE
./rand 1 2 1 3 | dd of=${BASEDIR}/randc_seed1_prob1_and_x0_x42_x100.data bs=1024 count=$SIZE
./rand 1 2 1 4 | dd of=${BASEDIR}/randc_seed1_prob1_and_x0_x42_x100_x127.data bs=1024 count=$SIZE
./rand 1 2 1 5 | dd of=${BASEDIR}/randc_seed1_prob1_and_x0_x42_x100_x127_x91.data bs=1024 count=$SIZE


./rand 1 1 1024 1 | dd of=${BASEDIR}/randc_seed1_prob1024_x0.data bs=1024 count=$SIZE
./rand 1 1 1024 2 | dd of=${BASEDIR}/randc_seed1_prob1024_xor_x0_x42.data bs=1024 count=$SIZE
./rand 1 1 1024 3 | dd of=${BASEDIR}/randc_seed1_prob1024_xor_x0_x42_x100.data bs=1024 count=$SIZE
./rand 1 1 1024 4 | dd of=${BASEDIR}/randc_seed1_prob1024_xor_x0_x42_x100_x127.data bs=1024 count=$SIZE
./rand 1 1 1024 5 | dd of=${BASEDIR}/randc_seed1_prob1024_xor_x0_x42_x100_x127_x91.data bs=1024 count=$SIZE

./rand 1 2 1024 2 | dd of=${BASEDIR}/randc_seed1_prob1024_and_x0_x42.data bs=1024 count=$SIZE
./rand 1 2 1024 3 | dd of=${BASEDIR}/randc_seed1_prob1024_and_x0_x42_x100.data bs=1024 count=$SIZE
./rand 1 2 1024 4 | dd of=${BASEDIR}/randc_seed1_prob1024_and_x0_x42_x100_x127.data bs=1024 count=$SIZE
./rand 1 2 1024 5 | dd of=${BASEDIR}/randc_seed1_prob1024_and_x0_x42_x100_x127_x91.data bs=1024 count=$SIZE


