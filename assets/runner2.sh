#!/bin/bash

# set a handler to clean the SCRATCHDIR once finished
trap "clean_scratch" TERM EXIT

cd /storage/brno3-cerit/home/ph4r05/booltest
/storage/brno3-cerit/home/ph4r05/.pyenv/versions/3.7.1/bin/python \
	  -m booltest.job_client \
	    --cwd=/storage/brno3-cerit/home/ph4r05/booltest/assets \
	      --logdir=/storage/brno3-cerit/home/ph4r05/bool-log \
	      --threads 1 --server wsexp.deadcode.me --port 4688 --time $((60*60*2)) \
	      --epoch 1 --delete-on-success 1 --log-to-scratch 1
