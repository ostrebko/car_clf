import os
import numpy as np
from utils.functions_with_keras import load_image



def create_prediction(img_path, config, model):
    
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
    new_image = load_image(img_path, config)

    # check prediction
    pred = model.predict(new_image)

    class_num = np.argmax(pred, axis=1)[0]
    class_name = config.class_names[class_num]

    print(f'for image {os.path.basename(img_path)} ' 
          f'predicted class and class_name: {(class_num, class_name)}')
    #return class_num, class_name




