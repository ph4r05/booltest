#!/usr/bin/env bash

# Old:
# export HOMEDIR="/storage/brno3-cerit/home/${LOGNAME}"
# rsync -av --exclude __pycache__ "${HOMEDIR}/booltest/booltest/" "${HOMEDIR}/.pyenv/versions/3.6.4/lib/python3.6/site-packages/booltest/"

# Unused:
# brno5-archive plzen2-archive plzen3-kky jihlava2-archive
export CENTRES=(brno1 brno2 brno3-cerit brno6 brno7-cerit brno8 brno9-ceitec budejovice1 \
jihlava1-cerit liberec1-tul ostrava1 plzen1 praha1 praha4-fzu praha5-elixir)

export HOMEDIR="/storage/brno3-cerit/home/${LOGNAME}"
export BOOLTEST="${HOMEDIR}/booltest"
export EACIRC_ESTREAM="${HOMEDIR}/eacirc-streams/build/eacirc-streams"
export PYENV_ROOT_CENTRAL="${HOMEDIR}/.pyenv"

for CCENTRE in ${CENTRES[@]}; do
    CFHOME="/storage/${CCENTRE}/home/${LOGNAME}"
    PYENV_CUR="${CFHOME}/.pyenv"
    BASHRCP="${CFHOME}/.bashrc"
    if [ ! -d "${CFHOME}" ]; then
        continue
    fi

    echo "===================================="
    echo "Centre: $CCENTRE, pyenv: ${PYENV_CUR}"
    rsync -av --delete --exclude __pycache__  "${BOOLTEST}/booltest/" "${PYENV_CUR}/versions/3.7.1/lib/python3.7/site-packages/booltest/"
done

