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

LOGDIR=~/testbed-proc/
mkdir -p ${LOGDIR}

# Clean previous tmp files
/bin/rm -rf /tmp/testbed-*

# Start processes
for cur in `seq 0 $((${NUMPROC} - 1))`;
do
    nice -n 19 taskset -c 31-63 nohup \
    python /home/xklinec/poly-verif/polyverif/testbed.py \
        --generator-path /home/xklinec/eacirc/build/generator/generator \
        --result-dir /home/xklinec/testbed-results \
        --tests-manuals ${NUMPROC} --tests-stride ${cur} \
        --matrix-size 1000 --matrix-comb-deg 1 2 3 \
        --top 128 --no-comb-and --only-top-comb --only-top-deg \
        --no-term-map --topterm-heap --topterm-heap-k 256 \
        >  ${LOGDIR}/testbed_t${TIMESTART}_n${NUMPROC}_c${cur}.out \
        2> ${LOGDIR}/testbed_t${TIMESTART}_n${NUMPROC}_c${cur}.err \
        &
    echo "Process ${cur}/$((${NUMPROC} - 1)) started"
    sleep 2
done

