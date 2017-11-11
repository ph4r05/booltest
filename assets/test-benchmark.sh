#!/bin/bash
TIMESTART=`date -u +%s`
NUMPROC=$1
NUMPROC=${NUMPROC:-4}
echo "Going to start computation with ${NUMPROC} processes. ID=${TIMESTART}"

LOGDIR=~/testbed-proc/
mkdir -p ${LOGDIR}

# Clean previous tmp files
/bin/rm -rf /tmp/testbed-*

# Start processes
for cur in `seq 0 $((${NUMPROC} - 1))`;
do
    nice -n 2 nohup \
    python ~/poly-verif/booltest/testbed.py \
        --generator-path ~/eacirc/generator/generator \
        --result-dir ~/testbed-results \
        --tests-manuals ${NUMPROC} --tests-stride ${cur} \
        --matrix-comb-deg 1 2 --matrix-deg 3 \
        --top 128 --no-comb-and --only-top-comb --only-top-deg \
        --no-term-map --topterm-heap --topterm-heap-k 256 \
        >  ${LOGDIR}/testbed_t${TIMESTART}_n${NUMPROC}_c${cur}.out \
        2> ${LOGDIR}/testbed_t${TIMESTART}_n${NUMPROC}_c${cur}.err \
        &
    echo "Process ${cur}/$((${NUMPROC} - 1)) started"
    sleep 2
done

