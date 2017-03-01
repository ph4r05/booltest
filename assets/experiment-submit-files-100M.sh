#!/bin/bash

for FL in "$@"
do
    BS=${FL##*/}
    echo "Submitting file $FL"

    for MB in 1 10 100
    do
        echo "$MB"
        submit-experiment -e ph4r05@gmail.com -n "${BS}_${MB}M" -c "/home/sample-configs/${MB}MB.json" -f "${FL}" -a
    done
done





