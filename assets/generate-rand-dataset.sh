#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 generator filesnum destdir"
    exit 1
fi


GENERATOR=$1
NUMFILES=$2
DESTDIR=$3
SIZEKB=112400

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
GENDIR="${DIR}/rndgen-${GENERATOR}"

if [ ! -d "$GENDIR" ]; then
    echo "Error, generator not found: $GENDIR"
    exit 1
fi

mkdir -p "${DESTDIR}"
for i in `seq 1 "$NUMFILES"`; do
    SEED=`python -c 'import random; print(random.randint(0,2**32-1))'`
    echo "Generating $i seed=$SEED"
    ${GENDIR}/rand $SEED | dd of="${DESTDIR}/batterytest_rand${GENERATOR}_${i}_seed${SEED}.data" bs=1024 count="${SIZEKB}"
    echo "Done $i"
done
