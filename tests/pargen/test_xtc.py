import json 
from pygama.pargen.xtc import prepare_baseline

with open("file_test_p08.json", "r") as f:
    data = json.load(f)

hit_files = data["hit"]
dsp_files = data["dsp"]
chn_id = data["chn_id"]

config = {
    "baseline_conditions": {"is_baseline": 63}
}

print(chn_id)

result = prepare_baseline(
    hit_files=hit_files,
    dsp_files=dsp_files,
    chn_id=chn_id[7],
    config=config,
)

print(result)
