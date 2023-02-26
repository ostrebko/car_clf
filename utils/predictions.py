import numpy as np
from utils.functions_with_keras import create_model, load_image


def create_prediction(img_path, config):
    
    # load model
    model = create_model(config)

    # load a single image
    new_image = load_image(img_path, config)

    # check prediction
    pred = model.predict(new_image)

    class_num = np.argmax(pred, axis=1)[0]
    class_name = config.class_names[class_num]

    return class_num, class_name




