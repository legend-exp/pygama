first: load container (singularity shell -B /mnt:/mnt legendexp_legend-software_latest_20190525154247.sqsh)

container is in /mnt/e15/schwarz/legend_cont

second: export DATADIR --> export DATADIR=/mnt/e15/schwarz/software/pygama/pygama/experiments




if python3 is installed in default position:
./process_test.py -t0 --verbose -r 204
seems to work somehow...

anyway, you can use:
python3 process_test.py -t0 --verbose -r <runNr>


good octal dump: use od -A x -t x1z -v


