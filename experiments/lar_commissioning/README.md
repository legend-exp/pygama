# lar-commissioning

Install pygama locally:
```console
$ cd software/pygama
$ ./legend-env.sh
$ cd src/pyfcutils && python -m pip install . && cd ..
$ cd src/pygama && python -m pip install . && cd ..
```

Configure files in `software/meta` and eventually edit `run-production.sh`, then:
```console
$ ./software/run-production.sh
```
To run the analysis pipeline.
