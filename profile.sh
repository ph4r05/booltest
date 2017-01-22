#!/bin/bash
python -m cProfile   polyverif/main.py data.rc --degree 3 --block 128 --tv 100Mi --top 100 | tee proftxt4
