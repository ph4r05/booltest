#!/bin/bash

export HOMEDIR="/storage/brno3-cerit/home/${LOGNAME}"
export DEBVER=`cat /etc/debian_version 2>/dev/null | cut -d '.' -f 1`
export EACIRC_ESTREAM="${HOMEDIR}/crypto-streams-v3.0"
#export EACIRC_ESTREAM="${HOMEDIR}/eacirc-streams-v2.3"
#export EACIRC_ESTREAM="${HOMEDIR}/eacirc-streams-v3"
#export EACIRC_ESTREAM="${HOMEDIR}/eacirc-streams/build/eacirc-streams-deb${DEBVER}"

module add gcc-5.3.0 2>/dev/null >/dev/null
module add cmake-3.6.1 2>/dev/null >/dev/null
module add mpc-1.0.3 2>/dev/null >/dev/null
module add gmp-6.1.2 2>/dev/null >/dev/null
module add mpfr-3.1.4 2>/dev/null >/dev/null

exec ${EACIRC_ESTREAM} $@
