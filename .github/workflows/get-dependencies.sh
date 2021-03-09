#!/bin/bash

pyver=$(python --version | sed 's|Python \([0-9]*\).\([0-9]*\).[0-9]*|\1\2|')

if [ "$pyver" -gt "35" ]; then
    sudo apt-get install -y llvm
else
    sudo apt-get install -y llvm-8
fi

python -m pip install --upgrade pip numpy
