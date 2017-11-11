#!/bin/bash

TOPK=128
TESTS=1000

for DATA in $((1024*1024*1)) $((1024*1024*10)) $((1024*1024*100)); do
for block in 128 256; do
    for deg in 1 2 3; do
        for k in 1 2 3; do
            echo "===================================================================================="
            echo "Computing block: $block, deg: $deg, k: $k, date: `date`"
            python ~/poly-verif/booltest/randverif.py --block ${block} --deg ${deg} --rounds 0 \
                    --tv ${DATA} --combine-deg ${k} --top ${TOPK} --test-aes  --tests ${TESTS} \
                    --csv-zscore --no-comb-and --only-top-comb --only-top-deg --no-term-map --topterm-heap --topterm-heap-k 256 \
                    > ~/aessize-${block}bl-${deg}deg-${k}k-${DATA}B-${TESTS}tests.out \
                    2> ~/aessize-${block}bl-${deg}deg-${k}k-${DATA}B-${TESTS}tests.err

            echo "*** Finished Computing block: $block, deg: $deg, k: $k, date: `date`"
        done
    done
done
done
echo "Done `date`"

