#!/bin/bash

[[ -f /etc/os-release ]] && linux_distro=$(awk -F= '/^NAME/{print $2}' /etc/os-release | cut -d ' ' -f1)

if [[ "${linux_distro}" == "Ubuntu" || "${linux_distro}" == "Debian" ]]; then

    pyver=$(python --version | sed 's|Python \([0-9]*\).\([0-9]*\).[0-9]*|\1\2|')

    if [ "$pyver" -gt "35" ]; then
        sudo apt-get install -y llvm
    else
        sudo apt-get install -y llvm-8
        sudo ln -s /usr/bin/llvm-config-8 /usr/bin/llvm-config
    fi
fi

python -m pip install --upgrade pip wheel setuptools numpy GitPython pytest
