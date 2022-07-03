### get the test data file:
```
on NERSC, file is located at: `/global/project/projectdirs/legend/data/test_data`
copy data file into: `~/Data/LNGS`
`bzip2 -d 2019-3-18-BackgroundRun204.bz2`
`mkdir pygama`
```

optional:
change Docker to run on 4 CPUs (whale icon/Preferences/Advanced)

Check what Docker image you installed:
```
docker image ls
```
You may need to change the startup line below (`legendexp/legend-software`) to your own image if it is different.

### start the container, (re)install pygama, and set DATADIR
NOTE: this reinstallation step is temporary, we will fix this soon
https://github.com/legend-exp/legendexp_legend-base_img

```
docker run -it --rm \
    -v "$HOME":"/home/user" \
    -v "$HOME/legend-base":"/root":delegated \
    -p 8888:8888 \
    -p 14500:14500 \
    legendexp/legend-software
pip uninstall pygama
cd /home/user
git clone https://github.com/legend-exp/pygama.git
pip install -e pygama
python
from pygama import DataSet
(ctrl-c)

export DATADIR=/home/user/Data
```

### start the tutorial:
jupyter lab --ip 0.0.0.0 --port 8888 --allow-root --no-browser
(open URL in browser)
(open pygama/experiments/teststand/lngs_tutorial.ipynb)
