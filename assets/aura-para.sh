#!/bin/bash
module add mpc-0.8.2
module add gmp-4.3.2
module add mpfr-3.0.0
module add cmake-3.6.2
export PATH=~/local/gcc-5.2.0/bin:$PATH
export LD_LIBRARY_PATH=~/local/gcc-5.2.0/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=~/local/gcc-5.2.0/lib64:$LD_LIBRARY_PATH

TIMESTART=`date -u +%s`
NUMPROC=$1
NUMPROC=${NUMPROC:-16}
echo "Going to start computation with ${NUMPROC} processes. ID=${TIMESTART}"

# Clean previous tmp files
/bin/rm -rf /tmp/testbed-*

LOGDIR=/tmp/testbed-proc
mkdir -p ${LOGDIR}

# Start processes
for cur in `seq 0 $((${NUMPROC} - 1))`;
do
    nice -n 15 taskset -c 30-63 nohup \
    python ~/poly-verif/booltest/testbed.py \
        --generator-path ~/eacirc/build/generator/generator \
        --result-dir ~/testbed-results \
        --data-dir /tmp/testdata \
        --tests-manuals ${NUMPROC} --tests-stride ${cur} --tests-random-select-seed 2 \
        --matrix-size 1 10 100 --matrix-comb-deg 1 2 3 --matrix-deg 1 2 \
        --top 128 --no-comb-and --only-top-comb --only-top-deg \
        --no-term-map --topterm-heap --topterm-heap-k 256 \
        >  ${LOGDIR}/testbed_t${TIMESTART}_n${NUMPROC}_c${cur}.out \
        2> ${LOGDIR}/testbed_t${TIMESTART}_n${NUMPROC}_c${cur}.err \
        &
    echo "Process ${cur}/$((${NUMPROC} - 1)) started"
    sleep 1
done

