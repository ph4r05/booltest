#!/bin/bash

JOB_START=$1
JOB_END=$2

if [[ ( -z "$JOB_START" ) || ( -z "$JOB_END" ) ]]; then
    echo "Usage: $0 job_start job_end"
    exit 1
fi

BATCH=50
for (( c=JOB_START; c<=JOB_END; c+=BATCH )); do
    AGG=""
    AGG_IDS=""
    for (( i=c; i< (c+$BATCH) && i <= JOB_END; i++ )); do
        AGG_IDS="$AGG_IDS $i"
        AGG="$AGG ${i}.arien-pro.ics.muni.cz"
    done
    echo "Canceling jobs $AGG_IDS"
    qdel $AGG
done

#for i in `seq $JOB_START $JOB_END`; do
#    echo "Canceling job $i"
#    qdel "${i}.arien-pro.ics.muni.cz"
#done





