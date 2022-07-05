This is an example of how to produce HADES char data with pygama.

Run processTiers.py with at least following options (See the file for details):

 -ds -> Number of your run
 -r  -> Also number of the run (redundant right now but needed if you want to analyse different runs at the same time)
 -t0 -> if you want to create t1 files (takes t0)
 -t1 -> if you want to create t2 files
 -s  -> Which subfile you want (we have 5 for HV scans, this should be reworked in the future)
 -db -> /pth/to/your/ rundb.json - Just the Path, you can run this from everywhere (All needed directories have to be defined in        the runDB) 


An example: processTiers.py -ds 21 -r 21 -t0 -t1 -s 1 -db /path/to/ 
   
