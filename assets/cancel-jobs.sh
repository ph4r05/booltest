#!/bin/bash

JOB_START=$1
JOB_END=$2

if [[ ( -z "$JOB_START" ) || ( -z "$JOB_END" ) ]]; then
    echo "Usage: $0 job_start job_end"
    exit 1
fi

for i in `seq $JOB_START $JOB_END`; do
    echo "Canceling job $i"
    qdel "${i}.arien-pro.ics.muni.cz"
done





