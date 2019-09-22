#!/bin/bash

# brno3-cerit brno5-archive plzen2-archive plzen3-kky jihlava2-archive
export CENTRES=(brno1 brno2 brno6 brno7-cerit brno8 brno9-ceitec budejovice1 \
jihlava1-cerit liberec1-tul ostrava1 plzen1 praha1 praha4-fzu praha5-elixir)


for CCENTRE in ${CENTRES[@]}; do
    CFHOME="/storage/${CCENTRE}/home/${LOGNAME}"

    BASHRCP="${CFHOME}/.bashrc"
    if [ ! -d "${CFHOME}" ]; then
        continue
    fi

    echo "===================================="
    echo "Centre: $CCENTRE"

    echo "Installing to ${CFHOME}..."
    export PYENV_ROOT="${CFHOME}/.pyenv"
    export PATH="${PYENV_ROOT}/bin:${PATH}"

    if [ ! -d "${CFHOME}/.pyenv" ]; then
        git clone https://github.com/pyenv/pyenv.git "${CFHOME}/.pyenv"
    else
        cd "${CFHOME}/.pyenv"
        git pull
    fi

    eval "$(pyenv init -)"
    pyenv install 2.7.14
    pyenv install 3.7.1
done



