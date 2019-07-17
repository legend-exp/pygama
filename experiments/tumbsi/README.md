*** FLASHCAM file reader ***

# File structure:
- Tier0 FlashCam file in ./Raw/Run0/
- Tier1 hdf5 file     in ./tier/
- Tier2 hdf5 file     in ./tier/

# Tier1 production
./process_test.py -ds 0 -r 000 --tier0 -o -v -n 1000000

# Tier2 production
./process_test.py -ds 0 -r 000 --tier1 -o -v -m -n 1000000

# Plot traces and energy spectra
open 'jupyter notebook'
run 'analysis.ipynb'

# Perform two-steps energy calibration + PSA
run 'python calibration.py -ds 0 -r 000 -db -p1 -p2 -sc'
run 'python fit_calibrated_peaks.py 0'
run 'python AvsE.py -ds 0'
