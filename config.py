import yaml
#loads config 
config_path="configs/config_mbs.yaml"
with open(config_path) as f:
    cfg = yaml.safe_load(f)