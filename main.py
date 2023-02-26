from utils.predictions import create_prediction
from utils.read_config import config_reader
from utils.functions import get_rnd_test_image, create_paths


if __name__ == "__main__":
    
    config = config_reader('config/data_config.json')
    config.IMG_SIZE = 448
    paths = create_paths(config, is_notebook=False)

    # image path
    random_img_path = get_rnd_test_image(paths)
    prediction = create_prediction(random_img_path, config)
    print(f'predicted class name: {prediction}')