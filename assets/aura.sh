#!/bin/bash
module add mpc-0.8.2
module add gmp-4.3.2
module add mpfr-3.0.0
module add cmake-3.6.2
export PATH=~/local/gcc-5.2.0/bin:$PATH
export LD_LIBRARY_PATH=~/local/gcc-5.2.0/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=~/local/gcc-5.2.0/lib64:$LD_LIBRARY_PATH

nice -n 19 python ~/poly-verif/booltest/testbed.py \
    --generator-path ~/eacirc/build/generator/generator \
    --result-dir ~/testbed-results \
    --tests-manuals 10 --tests-stride $1 \
    --top 128 --no-comb-and --only-top-comb --only-top-deg --no-term-map \
    --topterm-heap --topterm-heap-k 256 --best-x-combs 128


