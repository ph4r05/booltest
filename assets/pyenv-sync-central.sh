#!/bin/bash

# brno3-cerit brno5-archive plzen2-archive plzen3-kky jihlava2-archive
export CENTRES=(brno1 brno2 brno6 brno7-cerit brno8 brno9-ceitec budejovice1 \
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
    echo "Centre: $CCENTRE"
    rsync -av --delete "${PYENV_ROOT_CENTRAL}/" "${PYENV_CUR}/"
done



