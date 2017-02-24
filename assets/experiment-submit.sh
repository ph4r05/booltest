#!/bin/bash

FILE=$1
FNAME="${FILE}.data"

if [ ! -f $FNAME ]; then
    echo "Error, file $FNAME" not found
    exit 1
fi

for MB in 1 10 50 100 250 500 1000
do
    submit-experiment -e ph4r05@gmail.com -n "${FILE}_${MB}M" -c "/home/sample-configs/${MB}MB.json" -f "${FNAME}" -a
done




