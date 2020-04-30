#!/bin/bash

# Old:
export HOMEDIR="/storage/brno3-cerit/home/${LOGNAME}"
rsync -av --progress --exclude __pycache__ --exclude '*.pyc' "${HOMEDIR}/booltest/booltest/" "${HOMEDIR}/.pyenv/versions/3.7.1/lib/python3.7/site-packages/booltest/"

