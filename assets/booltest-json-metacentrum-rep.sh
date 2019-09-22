#!/bin/bash

export HOMEDIR="/storage/brno3-cerit/home/${LOGNAME}"
export BOOLTEST="${HOMEDIR}/booltest"
export EACIRC_ESTREAM="${HOMEDIR}/eacirc-streams/build/eacirc-streams"
cd $HOMEDIR

export MPICH_NEMESIS_NETMOD=tcp
export OMP_NUM_THREADS=$PBS_NUM_PPN
export PYENV_ROOT="${HOMEDIR}/.pyenv"
export PATH="${PYENV_ROOT}/bin:${PATH}"

C_ITER=0
RETCODE=2
TIME_START=$SECONDS

while (( $C_ITER < 6 && ($RETCODE == 2 || $RETCODE == 1) )); do

    C_ITER=$((C_ITER+1))
    echo "`hostname` starting ${C_ITER}..."

    TIME_ELAPSED=$(($SECONDS - $TIME_START))
    if (( $TIME_ELAPSED > 600 )); then
        echo "Elapsed time too big: ${TIME_ELAPSED}, quitting"
        break
    fi

    module add gcc-5.3.0
    module add cmake-3.6.1
    module add mpc-1.0.3
    module add gmp-6.1.2
    module add mpfr-3.1.4

    eval "$(pyenv init -)"
    RETCODE=$?
    if (( $RETCODE == 2 || $RETCODE == 1 )); then
        continue
    fi

    sleep 1

    pyenv local 3.7.1
    RETCODE=$?
    if (( $RETCODE == 2 || $RETCODE == 1 )); then
        continue
    fi
    sleep 1

    echo stdbuf -eL python -m booltest.booltest_json $@
    stdbuf -eL python -m booltest.booltest_json $@
    RETCODE=$?

done

exit $RETCODE
