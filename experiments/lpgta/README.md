# Post-GERDA Test

The scripts required to process and analysis the data collected during the Post-GERDA Test.

## Usage

When presented with a calibration data set, first set the location of the raw files to process under `raw_files` in `template.json`.  This list accepts environment variables and wildcards.  For example:

```json
"files_raw": [
    "${DATADIR}/lngs/pgt/v01.01/raw/geds/LPGTA_r0033_*_calib_geds_raw.lh5"
]
```

The data set is analyzed by executing the following:

```bash
python calibration.py -n
./submit-slurm.sh N "pz_preliminary.py -s produce"
python pz_preliminary.py -s analyze 1
./submit-slurm.sh N "processing.py -f cal -p 1"
python dqc.py -f cal
./submit-slurm.sh N "scatter.py -s produce"
python scatter.py -s analyze
./submit-slurm.sh N "pz_time.py -s produce"
python pz_time.py -s analyze
./submit-slurm.sh N "pz_average.py -s produce"
python pz_average.py -s analyze
./submit-slurm.sh N "processing.py -f opt -p 1"
python optimization.py -c N
./submit-slurm.sh N "processing.py -f dsp -p 1"
python dqc.py -f dsp
./submit-slurm.sh N "resolution.py -s produce"
python resolution.py -s analyze -c N
./submit-slurm.sh N "pz_energy.py -s produce"
python pz_energy.py -s analyze
./submit-slurm.sh N "noise.py -s produce"
python noise.py -s analyze
./submit-slurm.sh N "crosstalk.py -s produce"
python crosstalk.py -s analyze
```

where `N` is the number of parallel processes to launch, which, for the fastest processing, should be set to the number of channels to be processed.  The command-line arguments available are common to all of the scripts used and can be displayed by running one of them with the `-h` argument.  For example:

```bash
$ python processing.py -h
usage: processing.py [-h] [-n] [-d] [-o O] [-s S] [-f F] [-p P] [-c C] [-i I [I ...]] [-g G [G ...]]

optional arguments:
  -h, --help    show this help message and exit
  -n            create a new database file
  -d            run in debug mode
  -o O          set the output directory
  -s S          set the analysis stage
  -f F          set the processing label
  -p P          set the number of poles
  -c C          set the number of cores
  -i I [I ...]  set the indices to process
  -g G [G ...]  set the geds to process
```

Note that the `python` commands should be run inside the `legend-software` shifter image while the `./submit-slurm.sh` ones should not.  The latter commands submit one job per raw file to a Cori node at NERSC.  A log file per job is written to an automatically created directory in the user's scratech area based on the submission date and time, whose full path is outputted by `./submit-slurm.sh`.  Alternatively, such commands can be run locally.  For example:

```bash
python processing.py -f cal -p 1 -c N
```

## Output

The output files will be written to the directories specified in `template.json`.  In total, the output file size is about six percent of the raw file size, of which about 70 percent is in the `opt_dir` path.  It is recommended to set the output directories to a scratch area if the output file size is expected to be too large, such as more than the user's quota.

## References

Additional details can be found at:

- [Analysis Call, 07/08/21](https://indico.legend-exp.org/event/637/contributions/3078/attachments/1726/2653/sweigart_update.pdf)
- [Analysis Call, 08/05/21](https://indico.legend-exp.org/event/654/contributions/3164/attachments/1758/2717/sweigart_update.pdf)
