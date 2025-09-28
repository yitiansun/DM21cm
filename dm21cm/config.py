import os
import yaml


CONFIG_PATH = os.path.join(os.path.dirname(__file__), os.pardir, "config.yaml")

def load_config(config_path=CONFIG_PATH):
    with open(config_path) as f:
        _cfg = yaml.safe_load(f)

    config = {}
    for k, v in _cfg.items():
        if isinstance(v, str):
            v = os.path.expanduser(os.path.expandvars(v))
            if not os.path.isabs(v):
                # interpret relative to the config.yaml location
                v = os.path.abspath(os.path.join(os.path.dirname(config_path), v))
        config[k] = os.path.normpath(v)

    return config

CONFIG = load_config()