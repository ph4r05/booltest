#!/usr/bin/env bash
cd /storage/brno3-cerit/home/ph4r05/booltest
/storage/brno3-cerit/home/ph4r05/.pyenv/versions/3.7.1/bin/python \
    -m booltest.job_server --checkpoint chk3.json --rand-jobs --continuous-loading 1 ../b4/batcher-jobs-1588091902.json
