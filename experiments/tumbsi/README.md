*** FLASHCAM file reader ***

# File structure:
- Tier0 FlashCam file in ./Raw/Run0/
- Tier1 hdf5 file     in ./tier/
- Tier2 hdf5 file     in ./tier/

# Tier1 production
./process_test.py -ds 0 -r 000 --daq_to_raw -o -v -n 1000000

# Tier2 production
./process_test.py -ds 0 -r 000 --raw_to_dsp -o -v -n 1000000

# Perform two-steps energy calibration + PSA
run 'python calibration.py -ds 0 -r 000 -db -p1 -p2 -sc'
run 'python PSA.py -ds 0'
