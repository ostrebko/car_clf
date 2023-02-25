import numpy as np
import os
from utils.functions import create_model, load_image, select_class


def create_prediction(img_path, path_to_config='config/data_config.json'):
    
    # load model
    model = create_model(path_to_config)

    # load a single image
    new_image = load_image(img_path)

    # check prediction
    pred = model.predict(new_image)

    class_num = np.argmax(pred, axis=1)[0]
    class_name = select_class(class_num, path_to_config)

    return class_name




