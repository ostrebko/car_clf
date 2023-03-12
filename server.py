import os
import io
import base64
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
import keras.utils as image

# Imports of local libs
from utils.read_config import config_reader
from utils.functions import create_paths
from utils.functions_with_keras import create_model
from utils.predictions import create_prediction

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
config = config_reader('config/data_config.json')
paths = create_paths(config, is_not_in_root=False)
model = create_model(config, is_choice_by_input=False)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])

def predict(model=model):
    
    config.IMG_SIZE = 448
    # get the base64 encoded string
    im_b64 = request.json['im_b64']  #request.json.get('im_b64') 
    # convert it into bytes  
    img_bytes = base64.b64decode(im_b64.encode('utf-8'))
    # convert bytes data to PIL Image object
    img = Image.open(io.BytesIO(img_bytes))
    img = img.resize((config.IMG_SIZE, config.IMG_SIZE), Image.Resampling.LANCZOS)

    # PIL image object to numpy array
    img = image.img_to_array(img)
    img_arr = np.expand_dims(img, axis=0)
    img_arr /= 255.
    
    class_num, class_name = create_prediction(img_arr, config, model)
    #print(f'Send to client predict with class_num: {class_num}, class_name: {class_name}')
    
    return jsonify({
        "class_num": str(class_num),
        "class_name": str(class_name),
    })

    
if __name__ == '__main__':
    
    app.run('localhost', 5000)