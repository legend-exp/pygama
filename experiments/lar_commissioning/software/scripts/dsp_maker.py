from pygama.io.raw_to_dsp import raw_to_dsp

home_dir = "/home/pkrause/software/pygama/experiments/lar_commissioning/"
file_raw = home_dir + "data/com/raw/2022-04-13-sipm-test/20220425-184658-source-8000-athr200,12,2-es4000_raw.lh5"
file_config = home_dir + "software/meta/com/sipmtest-202110/r2d_config_v2.json"
file_dsp = home_dir +"data/com/dsp/2022-04-13-sipm-test/20220425-184658-source-8000-athr200,12,2-es4000_small_dsp.lh5"

raw_to_dsp(f_raw=file_raw,f_dsp=file_dsp, dsp_config=file_config,n_max=50000)