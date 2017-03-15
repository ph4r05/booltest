#!/bin/bash

TOPK=128

python ~/poly-verif/polyverif/testbed.py \
    --generator-path ~/eacirc/generator/generator \
    --result-dir ~/testbed-results \
    --tests-manuals 2 --tests-stride 0 \
    --top ${TOPK} \
    --no-comb-and --only-top-comb --only-top-deg --no-term-map --topterm-heap --topterm-heap-k 256 --best-x-combs 128 \
    > ~/benchtest.out \
    2> ~/benchtest.err

