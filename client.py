import requests
from utils.functions import create_paths, get_path_image, encode_image
from utils.read_config import config_reader


if __name__ == '__main__':
    
    config = config_reader('config/data_config.json')
    paths = create_paths(config, is_in_root=True)
    img_path = get_path_image(paths, is_work_demonstrate=False)
    
    im_b64_str = encode_image(img_path)
    
    r = requests.post('http://localhost:5000/predict', json={"im_b64": im_b64_str})
    
    if r.status_code == 200:
        print(f"For image: {img_path} classification prediction: ", 
              f"class_num: {r.json()['class_num']}, class_name:{r.json()['class_name']}")
    else:
        print(r.status_code, 'Check your request')