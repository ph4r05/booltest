#!/bin/bash

export HOMEDIR="/storage/brno3-cerit/home/${LOGNAME}"
export BOOLTEST="${HOMEDIR}/booltest"
export EACIRC_ESTREAM="${HOMEDIR}/eacirc-streams/build/eacirc-streams"
cd $HOMEDIR

export MPICH_NEMESIS_NETMOD=tcp
export OMP_NUM_THREADS=$PBS_NUM_PPN
export PYENV_ROOT="${HOMEDIR}/.pyenv"
export PATH="${PYENV_ROOT}/bin:${PATH}"
echo "`hostname` starting..."

module add gcc-5.3.0
module add cmake-3.6.1
module add mpc-1.0.3
module add gmp-6.1.2
module add mpfr-3.1.4

eval "$(pyenv init -)"
sleep 1

pyenv local 3.7.1
sleep 1

cd booltest

#exec stdbuf -eL python booltest/testjobsproc.py --out-dir ../bool-procs --narrow2 ../bool-res
exec stdbuf -eL python booltest/testjobsproc.py --out-dir ../bool-procs --pval-data=pval_db.json  ../bool-res-sec
