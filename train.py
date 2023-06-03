import argparse
import json
from src.configs import TrainConfig, DataConfig, ModelConfig
from src.utils import train, initialize_train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='./configs/default_config.json')

    args = parser.parse_args()

    with open(args.config) as config_file:
        configs = json.load(config_file)
    
    train_config = TrainConfig(**configs['train'])
    data_config = DataConfig(**configs['data'])
    model_config = ModelConfig(**configs['model'])

    train_params = initialize_train(train_config,data_config,model_config)
    train(**train_params)
