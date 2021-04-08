#!/bin/bash

pyver=$(python --version | sed 's|Python \([0-9]*\).\([0-9]*\).[0-9]*|\1\2|')

if [ "$pyver" -gt "35" ]; then
    sudo apt-get install -y llvm
else
    sudo apt-get install -y llvm-8
    sudo ln -s /usr/bin/llvm-config-8 /usr/bin/llvm-config
fi

python -m pip install --upgrade pip numpy
