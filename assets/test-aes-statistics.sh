#!/bin/bash

DATA=$((1024*1024*10))
TOPK=128
TESTS=100

for block in 128 256 384 512; do
    for deg in 1 2 3; do
        for k in 1 2 3; do
            echo "===================================================================================="
            echo "Computing block: $block, deg: $deg, k: $k, date: `date`"
            python ~/poly-verif/booltest/randverif.py --block ${block} --deg ${deg} --rounds 0 \
                    --tv ${DATA} --combine-deg ${k} --top ${TOPK} --test-aes  --tests ${TESTS} \
                    --csv-zscore --no-comb-and --only-top-comb --only-top-deg --no-term-map --topterm-heap --topterm-heap-k 256 \
                    > ~/aestest-${block}bl-${deg}deg-${k}k-${DATA}B-${TESTS}tests.out \
                    2> ~/aestest-${block}bl-${deg}deg-${k}k-${DATA}B-${TESTS}tests.err

            echo "*** Finished Computing block: $block, deg: $deg, k: $k, date: `date`"
        done
    done
done
echo "Done `date`"

