import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from flask import Flask, request, jsonify

# Imports of local libs
from utils.read_config import config_reader
from utils.functions import create_paths
from utils.functions_with_keras import create_model, decode_image
from utils.predictions import create_prediction

config = config_reader('config/data_config.json')      # load configs
paths = create_paths(config, is_in_root=True)          # load configs
model = create_model(config, is_choice_by_input=False) # load model


app = Flask(__name__)

@app.route('/predict', methods=['POST'])

def predict(model=model):
        
    config.IMG_SIZE = 448
    im_b64_str = request.json['im_b64'] # get the base64 encoded string
    img_arr = decode_image(im_b64_str, config)
    class_num, class_name = create_prediction(img_arr, config, model)
        
    return jsonify({
        "class_num": str(class_num), "class_name": str(class_name)})

    
if __name__ == '__main__':
    app.run('localhost', 5000)