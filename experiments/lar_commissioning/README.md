# lar-commissioning

Install pygama and pyfcutils locally:
```console
$ cd software/pygama-v01
$ mkdir src && cd src
$ git clone https://github.com/legend-exp/pygama
$ git clone https://github.com/legend-exp/pyfcutils
$ ./legend-env.sh
$ cd src/pyfcutils && python -m pip install . && cd ..
$ cd src/pygama && python -m pip install . && cd ..
```

Put a `legend-container.sif` file in `containers/`:
```console
$ cd software
$ mkdir containers && cd containers
$ ln -s /path/to/legendexp_legend-base_latest.sif legend-container.sif
```

Place (a symlink to) fcio files somewhere below `data/`, e.g.
```console
$ mkdir -p data/daq && cd data/daq
$ ln -s /path/to/daq .
```

This is how the directory tree should look at the end:
```
lar-commissioning
├── data
│   ├── daq
│   │   ├── pgt
│   │   └── ...
│   ├── dsp
│   │   ├── pgt
│   │   └── ...
│   └── raw
│       ├── pgt
│       └── ...
├── README.md
└── software
    ├── containers
    │   ├── legend-container.sif -> legendexp_legend-base_latest_20210503071451.sif
    │   └── legendexp_legend-base_latest_20210503071451.sif
    ├── meta
    │   ├── pgt
    │   └── ...
    ├── pygama-v01
    │   ├── legend-env.sh
    │   ├── local
    │   ├── pygama-run.py
    │   └── src
    ├── README
    └── run-production.sh
```

Configure files in `software/meta` and eventually edit `run-production.sh`, then:
```console
$ ./software/run-production.sh
```
To run the analysis pipeline.

For interactive analysis, start first a container instance:
```console
$ software/legend-env.sh
```
