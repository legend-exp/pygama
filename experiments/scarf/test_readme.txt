
need to export DATADIR beforehand, of course...
=======
first: load container (singularity shell -B /mnt:/mnt legendexp_legend-software_latest_20190525154247.sqsh)

container is in /mnt/e15/schwarz/legend_cont

second: export DATADIR --> export DATADIR=/mnt/e15/schwarz/software/pygama/pygama/experiments

modify the database json file (within there: put run number and description in data set)



if python3 is installed in default position:
./process_test.py -t0 --verbose -r 204
seems to work somehow...

anyway, you can use:
python3 process_test.py -t0 --verbose -r <runNr>


good octal dump: use od -A x -t x1z -v

viewing waveforms from a run:

python3 viewTier1.py <runNr>


