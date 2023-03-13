import base64
from dotmap import DotMap
from glob import glob
import matplotlib.pyplot as plt
import os
import random
from skimage import io
import zipfile




def create_paths(config, is_in_root=False):
    
    """
    Create paths to read, save and load data, models for train and inference
    
    -------
    params:
    
    config - dict (Dotmap) from configuration file with defined parameters values 
             (creates from config_reader function by reading data_config.json)
    is_in_root=True for main.py inference or run app .py in root dir or 
    is_in_root=False for notebook train
    
    """
    
    paths_dict = dict()

    if is_in_root:
        paths_dict['PATH_DATA'] = config.PATH_DATA
        paths_dict['PATH_MODELS'] = config.folder_models
    else: 
        paths_dict['PATH_DATA'] = os.path.join('..', config.PATH_DATA)
        paths_dict['PATH_MODELS'] = os.path.join('..', config.folder_models)

    paths_dict['PATH_DATA_TRAIN'] = os.path.join(paths_dict['PATH_DATA'], 
                                                 config.folder_data_train)
    paths_dict['PATH_OUTPUTS'] = os.path.join(paths_dict['PATH_DATA'], 
                                              config.folder_outputs)
    paths_dict['PATH_BEST_MODEL'] = os.path.join(paths_dict['PATH_DATA'], 
                                                 config.folder_best_model) 
    paths_dict['PATH_TO_ZIP'] = os.path.join(paths_dict['PATH_DATA'], 
                                             config.zip_file_name)
    paths_dict['PATH_TO_DATA_TRAIN'] = os.path.join(paths_dict['PATH_DATA_TRAIN'], 
                                                    config.folder_train_pics)
    paths_dict['PATH_TO_DATA_TEST'] = os.path.join(paths_dict['PATH_DATA_TRAIN'], 
                                                   config.folder_test_pics)
    paths_dict['PATH_TO_TEST_PREDICTIONS'] = os.path.join(paths_dict['PATH_DATA'], 
                                                          config.test_imgs_for_pred)
    paths = DotMap(paths_dict)
    
    return paths



def imshow(image_RGB):
    """
    Simple function to show image in RGB mode
    
    -------
    
    params:
    image_RGB - data from PIL.Image.open
    
    """
    io.imshow(image_RGB)
    io.show()



def extract_data_from_zip(path_to_big_zip, path_data_train, is_true=False):
    
    """
    Function for extracting data from zip-archive sf-dl-car-classification.zip
    and creates folders
    
    -------
    params:

    path_to_big_zip - path to sf-dl-car-classification.zip, defined in config
    path_data_train - path to unzip data, defined in config
    
    """

    if is_true:
        # Extract zip-archive with all data
        print('Unzip sf-dl-car-classification.zip')
        
        # Extract to folder 'inputs_for_train' without subfolder 'sf-dl-car-classification'
        with zipfile.ZipFile(path_to_big_zip) as z_file:
            for zip_info in z_file.infolist():
                if zip_info.filename[-1] == '/':
                    continue
                zip_info.filename = os.path.basename(zip_info.filename)
                z_file.extract(zip_info, path_data_train)

        
        # Unzip the files so that you can see them..
        print('Unzip pictures')
        for k, data_zip in enumerate(['train.zip', 'test.zip']):
            with zipfile.ZipFile(os.path.join(path_data_train, data_zip),"r") as z:
                z.extractall(path_data_train)
            os.remove(os.path.join(path_data_train, data_zip))

        print(os.listdir(path_data_train))

    else:
        print('Zip-archive "sf-dl-car-classification.zip" no need to unpack,',
              'to unpack change value of param "is_true" to "True"')



def plot_history(history, PATH_TO_SAVE, step_num, is_save_fig):
    
    """
    Function to print pictures of training model history in notebook 
    with possibility locally saving
    
    -------
    params:
    
    history - train hystory which created from model.fit
    PATH_BEST_MODEL - path to save fig if is_save_fig=True
    is_save_fig - True (save fig) or False (show fig)
    
    """
    
    plt.figure(figsize=(10,5))
    #plt.style.use('dark_background')
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    
    if is_save_fig:
        plt.savefig(os.path.join(PATH_TO_SAVE, f'Train_Vall_acc_st_{step_num}.png'))

    #plt.figure()
    plt.figure(figsize=(10,5))
    #plt.style.use('dark_background')
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    if is_save_fig:
        plt.savefig(os.path.join(PATH_TO_SAVE, f'Train_Vall_loss_st_{step_num}.png'))
    else:
        plt.show()


def get_label_map(paths_dict):
    
    """
    Function for creating label map dict 
    uses for submission to Kaggle.
    
    -------
    params:
    
    paths_dict - dict of paths which created from create_paths func
    
    """
    
    num_labels = map(os.path.basename, 
                     glob(os.path.join(paths_dict.PATH_TO_DATA_TRAIN, '*')))
    label_map_dict = {i : int(i) for i in num_labels}

    return label_map_dict



def get_path_image(paths_dict, is_work_demonstrate=True):
    
    """
    Function to image path:
    random path image from 'test_upload' folder or
    image path from 'handly' input
    
    -------
    params:
    
    paths_dict - dict of paths which created from create_paths func
    is_work_demonstrate=True - return random path
    is_work_demonstrate=False - ask to input img path, check inputed path and 
                                return it if image exists
    
    """
    
    if is_work_demonstrate:
        path_rnd_img = random.choice(glob(
            os.path.join(paths_dict.PATH_TO_TEST_PREDICTIONS, '*')))
    
    else:
        is_path_input = False
        while not is_path_input:
            path_rnd_img = input('Input full img path or relative img path in this project: ')
            is_path_input = os.path.exists(path_rnd_img)

    return path_rnd_img


def encode_image(img_path):
    with open(img_path, "rb") as f:
        im_bytes = f.read()     
    im_b64_str = base64.b64encode(im_bytes).decode("utf8")
    return im_b64_str