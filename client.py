import base64
import requests
from utils.functions import create_paths, get_path_image
from utils.read_config import config_reader


if __name__ == '__main__':
    
    config = config_reader('config/data_config.json')
    paths = create_paths(config, is_not_in_root=False)
    img_path = get_path_image(paths, is_work_demonstrate=False)
    
    with open(img_path, "rb") as f:
        im_bytes = f.read()     
    im_b64 = base64.b64encode(im_bytes).decode("utf8")
    
    r = requests.post('http://localhost:5000/predict', json={"im_b64": im_b64})
    
    if r.status_code == 200:
        print(f"For image: {img_path} predict classification: ", 
              f"class_num: {r.json()['class_num']}, class_name:{r.json()['class_name']}")
    else:
        print(r.status_code, 'проверьте Ваш запрос')