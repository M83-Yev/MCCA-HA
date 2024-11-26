import os
import yaml

def load_config(config_file="config.yaml"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, config_file)

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

CONFIG = load_config()


