#!/bin/bash
rsync -av --progress --exclude __pycache__ --exclude '*.pyc' ~/booltest/booltest/ ~/.pyenv/versions/3.7.1/lib/python3.7/site-packages/booltest/
