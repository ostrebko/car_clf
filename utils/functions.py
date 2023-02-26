from dotmap import DotMap
from glob import glob
import matplotlib.pyplot as plt
import os
import random
from skimage import io
import zipfile




def create_paths(config, is_notebook=True):
    
    """
    create paths for data: for notebook is_notebook=True or for main.py is_notebook=False
    params:
    config - config from class config_reader

    """
    
    paths_dict = dict()

    if is_notebook:
        paths_dict['PATH_DATA'] = os.path.join('..', config.PATH_DATA)
    else: 
        paths_dict['PATH_DATA'] = config.PATH_DATA

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
    paths = DotMap(paths_dict)
    
    return paths



def imshow(image_RGB):
    """
    Simple function to show image in RGB
    image_RGB from PIL.Image.open
    """
    io.imshow(image_RGB)
    io.show()



def extract_data_from_zip(path_to_big_zip, path_data_train, is_true=False):
    """
    function for extract data from zip-archive sf-dl-car-classification.zip
    and creates folders
    -------
    params:
    path_to_big_zip - path to sf-dl-car-classification.zip, defined in config
    path_data_train - path to unzip data, defined in config

    """

    if is_true:
        # Extract zip-archive with all data
        print('Unzip sf-dl-car-classification.zip')
        #with zipfile.ZipFile(path_to_big_zip,"r") as z:
        #    z.extractall(path_data_train)

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



def plot_history(history, PATH_BEST_MODEL, step_num, is_save_fig):
    
    """
    Function to print pictures of training model history in notebook 
    with possibility locally saving
    
    history - create from model.fit
        
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
        plt.savefig(os.path.join(PATH_BEST_MODEL, f'Train_Vall_acc_st_{step_num}.png'))

    #plt.figure()
    plt.figure(figsize=(10,5))
    #plt.style.use('dark_background')
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    if is_save_fig:
        plt.savefig(os.path.join(PATH_BEST_MODEL, f'Train_Vall_loss_st_{step_num}.png'))
    else:
        plt.show()


def get_label_map(paths_dict):
    
    num_labels = map(os.path.basename, glob(os.path.join(paths_dict.PATH_TO_DATA_TRAIN, '*')))
    label_map_dict = {i : int(i) for i in num_labels}

    return label_map_dict


def get_rnd_test_image(paths):
    rnd_img = random.choice(glob(os.path.join(paths.PATH_TO_DATA_TRAIN, '*', '*')))
    return rnd_img
