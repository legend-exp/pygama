*** FLASHCAM file reader ***

# File structure:
- Tier0 FlashCam file in ./Raw/Run0/
- Tier1 hdf5 file     in ./tier/
- Tier2 hdf5 file     in ./tier/

# Tier1 production
./process_test.py -ds 0 -r 000 --tier0 -o -v

# Tier2 production
./process_test.py -ds 0 -r 000 --tier1 -o -v -m

# Plot traces and energy spectra
open 'jupyter notebook'
run 'analysis.ipynb'
