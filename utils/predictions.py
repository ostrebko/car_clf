import os
import numpy as np
from utils.functions_with_keras import load_image
from utils.functions import create_paths, get_path_image
from utils.functions_with_keras import create_model




def create_prediction(new_image, config, model): #img_path
    
    """
    Function for creating prediction (classification) for one image.
    
    -------
    params:
    
    img_path - path for image
    config - dict (Dotmap) from configuration file with defined parameters values 
             (creates from config_reader function by reading data_config.json)
    model - compiled (and trained) Keras model to create prediction

    """

    # load a single image
    #new_image = load_image(img_path, config)

    # check prediction
    pred = model.predict(new_image)

    class_num = np.argmax(pred, axis=1)[0]
    class_name = config.class_names[class_num]

    return class_num, class_name



def make_predictions(config):
    
    """
    Function for creating predictions (classification) for images.
    
    -------
    params:
    
    config - dict (Dotmap) from configuration file with defined parameters values 
             (creates from config_reader function by reading data_config.json)
    
    param 'is_choice_by_input' in 'create_model' class makes it possible 
            to select a trained model

    """


    demonstrate_mode = config.demo[input('input 1 - to run in demonstrate mode, '
                                         '0 - to run with handle input image path: ' )]
    paths = create_paths(config, is_not_in_root=False)
    model = create_model(config, is_choice_by_input=False)

    while config.continue_predict:
        
        img_path = get_path_image(paths, is_work_demonstrate=demonstrate_mode)
        
        # load a single image
        new_image = load_image(img_path, config)

        class_num, class_name = create_prediction(new_image, config, model)

        print(f'for image {os.path.basename(img_path)} ' 
              f'predicted class and class_name: {(class_num, class_name)}')
        
        num_choise = input('input any num or simbol to continue predict, to exit - 0: ')
        if num_choise == '0':
            break
