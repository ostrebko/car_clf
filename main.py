from utils.read_config import config_reader
from utils.predictions import make_predictions


if __name__ == "__main__":
    
    config = config_reader('config/data_config.json')
    make_predictions(config)