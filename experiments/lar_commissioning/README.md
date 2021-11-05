# lar-commissioning

## Tutorials & documentation

First of all, read this:
* https://github.com/mmatteo/legend-analysis-tutorials
* https://github.com/legend-exp/pygama/blob/dev/tutorials/WaveformBrowserTutorial.ipynb
* https://github.com/legend-exp/pygama/blob/dev/tutorials/IntroToDSP.ipynb

## Setting up the environment

Clone pygama and make sure you're sitting in the right branch:
```console
$ git clone https://github.com/legend-exp/pygama
$ cd pygama && git checkout exp-lar-commissioning
```

Install pygama and pyfcutils in the container, for example:
```console
$ ./software/bin/legend-env.sh
$ cd /tmp
$ git clone https://github.com/legend-exp/pyfcutils
$ git clone https://github.com/legend-exp/pygama
$ cd pyfcutils && python -m pip install . && cd ..
$ cd pygama && git checkout exp-lar-commissioning && python -m pip install . && cd ..
```

Place (a symlink to) fcio files somewhere below `data/`, e.g.
```console
$ mkdir -p data/com && cd data/com
$ ln -s </path/to/raw> .
```

This is how the directory tree should look at the end (e.g. at MPIK):
```
lar_commissioning
├── data
│   ├── com
│   │   ├── daq -> /lfs/l1/legend/data/com/daq
│   │   │   ├── sipmtest-202110
│   │   │   └── ...
│   │   ├── raw -> /lfs/l1/legend/data/com/raw
│   │   │    └── sipmtest-202110
│   │   └── dsp
│   │        └── sipmtest-202110
│   └── pgt
│       ├── daq -> /lfs/l1/legend/data/pgt
│       │   ├── run0028-mid-june-sipm-test
│       │   └── ...
│       ├── raw
│       │   └── run0028-mid-june-sipm-test
│       └── dsp
│           └── run0028-mid-june-sipm-test
├── README.md
└── software
    ├── bin
    │   ├── legend-container.sif -> /lfs/l1/legend/software/legend-base.sif
    │   ├── legend-env.sh
    │   ├── pygama-run.py
    │   ├── run-production-pgt.sh
    │   └── run-production.sh
    └── meta
        ├── com
        │   ├── sipmtest-202108
        │   │   ├── d2r_config.json
        │   │   └── r2d_config.json
        │   └── sipmtest-202110
        │       ├── d2r_config.json
        │       └── r2d_config.json
        └── pgt
            └── run0028-mid-june-sipm-test
                ├── d2r_config.json
                └── r2d_config.json
```

Configure files in `software/meta` and eventually edit `software/bin/run-production.sh`, then:
```console
$ cd software/bin
$ ./legend-env.sh ./run-production.sh
```
To run the analysis pipeline inside the container.

For interactive analysis, you can work on a shell from within the container.
Just invoke `legend-env.sh` without any command line argument:
```console
$ software/bin/legend-env.sh
$ software/bin/legend-env.sh ipython
$ ...
```
