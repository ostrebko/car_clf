from utils.predictions import create_prediction
from utils.read_config import config_reader
from utils.functions import create_paths, get_path_image
from utils.functions_with_keras import create_model


if __name__ == "__main__":
    
    demonstrate_mode = True
    config = config_reader('config/data_config.json')
    paths = create_paths(config, is_notebook=False)
    model = create_model(config, is_choice_by_input=False)

    while config.continue_predict:
        img_path = get_path_image(paths, is_work_demonstrate=demonstrate_mode)
        create_prediction(img_path, config, model)
        
        num_choise = input('input any num or simbol to continue predict, to exit - 0: ')
        if num_choise == '0':
            break