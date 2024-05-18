import os

def read_config(file_path):
    config = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith('#'):
                key, value = line.split('=', 1)
                config[key.strip()] = value.strip()
    return config

# Read the configuration file
config_file_path = 'config.properties'  # Update this to the actual path of your config.properties file
config = read_config(config_file_path)

# Set the environment variable
os.environ['HF_DATASETS_CACHE'] = config['dataset_location']
from datasets import load_dataset

DATASET_NAME = "Cohere/wikipedia-2023-11-embed-multilingual-v3"
dataset = load_dataset(DATASET_NAME, 'en')
