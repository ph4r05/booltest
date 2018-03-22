#!/usr/bin/env bash

export HOMEDIR="/storage/brno3-cerit/home/${LOGNAME}"
rsync -av --exclude __pycache__ "${HOMEDIR}/booltest/booltest/" "${HOMEDIR}/.pyenv/versions/3.6.4/lib/python3.6/site-packages/booltest/"

