import base64
from dotmap import DotMap
from glob import glob
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


from tensorflow import keras
from keras import optimizers
import keras.utils as image
from PIL import Image
import io

from ImageDataAugmentor.image_data_augmentor import *
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau # LearningRateScheduler
from utils.read_config import config_reader
from utils.model import ModelForTrain



# Callbacks that used for training model
def callbacks(PATH_BEST_MODEL, config):                               

    """
    Function creates callback list from Keras callbacks, that used in fit method
    List consists 3 function: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
    (for a detailed description, see: https://keras.io/api/callbacks/)
    
    -------
    params:

    PATH_BEST_MODEL - path to save model with best params during model train
    config - dict (Dotmap) from configuration file with defined parameters values 
             (creates from config_reader function by reading data_config.json)

    """

    checkpoint = ModelCheckpoint(os.path.join(PATH_BEST_MODEL, config.best_model_name), 
                                 monitor=config.monitor_m_checkpnt, 
                                 verbose=config.verbose_m_checkpnt, 
                                 mode=config.mode_m_checkpnt, 
                                 save_best_only=config.save_best_only
                                 )

    earlystop = EarlyStopping(monitor=config.monitor_early_stop, 
                              patience=config.patience_early_stop, 
                              restore_best_weights=config.restore_best_weights
                              )

    reduce_lr = ReduceLROnPlateau(monitor=config.monitor_reduce_plteau, 
                                  factor=config.factor_reduce_plteau, 
                                  patience=config.patience_reduce_plteau, 
                                  verbose=config.verbose_reduce_plteau,
                                  min_lr=config.LR/config.min_lr_ratio
                                  )
    
    return [checkpoint, earlystop, reduce_lr]



def save_model(paths, config, model, step_num):
    
    """
    Function loads best saved weights to model after model training from PATH_BEST_MODEL,
    renames with num step and saves best weights and model to 'models' folder 

    -------
    params:

    PATH_BEST_MODEL - path to load weights with best params after training model
    config - dict (Dotmap) from configuration file with defined parameters values 
             (creates from config_reader function by reading data_config.json)
    model - model after one or few steps train
    step_num - number (int) of step train model (steps differs from each other
               with params, e.g.: numb of frozen layers img size, batch size e.t.c.)

    """

    # load best weights
    model.load_weights(os.path.join(paths.PATH_BEST_MODEL, config.best_model_name))
    
    # Save model & load best iteration after fitting (best_model):
    model.save(os.path.join(paths.PATH_MODELS, f'model_step_{step_num}.h5'))
    model.save_weights(os.path.join(paths.PATH_MODELS, f'weights_step_{step_num}.hdf5'))
    print(f'model and weights of {step_num} step traininig are saved in {paths.PATH_MODELSL}')



def eval_model(model, test_generator):
    
    """
    Function to evaluate the metric accuracy of trained or compiled model
    
    -------
    params:

    model - compiled Keras model for evaluating 
    test_generator - test generator (generator without augmentations)

    """
    
    scores = model.evaluate(test_generator, verbose=1)
    print("Accuracy: %.2f%%" % (scores[1]*100))



def make_predictions(test_sub_generator, model, label_map):

    """
    Function to make predictions (without using augmentations) 
    from folder 'test_upload' and using DataFrame from 'sample_submission.csv'
        
    -------
    params:

    test_sub_generator - generator which created with DataFrame from 'sample_submission.csv'
                         (uses photos from folder 'test_upload').
                         There were used AUGMENTATIONS=None when define this method from class.
                         (see class create_generator in generator.py)
    model - compiled (and trained) Keras model to create prediction
    label_map - dict with form {str(cls_numb): cls_numb} with all class numbs. 
                Uses for creation to Kaggle predict form

    """

    test_sub_generator.reset()
    pred = model.predict(test_sub_generator, steps=len(test_sub_generator),  verbose=1) 
    predictions = np.argmax(pred, axis=-1) #multiple categories
    label_map = dict((v,k) for k,v in label_map.items()) #flip k,v
    predictions = [label_map[k] for k in predictions]
    
    return predictions



def make_tta(config, test_sub_generator, model, label_map):
    
    """
    Function to make predictions with test time augmentations 
    (predictions with using augmentations) 
    from folder 'test_upload' and using DataFrame from 'sample_submission.csv'
        
    -------
    params:

    config - dict (Dotmap) from configuration file with defined parameters values 
             (creates from config_reader function by reading data_config.json)
    test_sub_generator - generator which created with DataFrame from 'sample_submission.csv'
                         (uses photos from folder 'test_upload').
                         There were used AUGMENTATIONS != None when define this method from class.
                         (see class create_generator in generator.py)
    model - compiled (and trained) Keras model to create prediction
    label_map - dict with form {str(cls_numb): cls_numb} with all class numbs. 
                Uses for creation to Kaggle predict form

    """

    tta_steps = config.steps_for_tta
    predictions_list = []

    for i in range(tta_steps):
        test_sub_generator.reset()
        preds = model.predict(test_sub_generator, steps=len(test_sub_generator),  verbose=1) 
        predictions_list.append(preds)
    pred = np.mean(predictions_list, axis=0)

    predictions_tta = np.argmax(pred, axis=-1) #multiple categories
    label_map = dict((v,k) for k,v in label_map.items()) #flip k,v
    predictions_tta = [label_map[k] for k in predictions_tta]

    return predictions_tta



def make_submission(filenames_with_dir, predictions):
    
    """
    Function to make submission for Kaggle competition
        
    -------
    params:

    filenames_with_dir - filenames from 'test_upload', creates from test_sub_generator.filenames
    predictions - predictions for images in folder 'test_upload'
                  with using trained model and test_sub_generator

    """

    submission = pd.DataFrame({'Id':filenames_with_dir, 'Category':predictions}, columns=['Id', 'Category'])
    submission['Id'] = submission['Id'].replace('test_upload/','')
    submission.to_csv('submission.csv', index=False)
    print('Save submit')
    
    return submission


def choice_model(is_choice_by_input):
    
    """
    Simple function for choosing name of trained model from 'models' folder.
    
    -------
    params:
    is_choice_by_input - boolean. If True - allows to choose and return
                         one of the model names from 'models' folder.
                         If false - return model name from last saved step 

    """

    d = {}
    for k, item in enumerate(glob(os.path.join('models', '*'))):
        d[k+1]=os.path.basename(item)
    
    if is_choice_by_input:
        print(f'trained models dict: {d}')
        select_model_num = int(input('Input model num: '))
    else:
        select_model_num = list(d.keys())[-1]
    path_model_name = os.path.join('models', d[select_model_num])
    
    return path_model_name



def create_model(config, is_choice_by_input):
    
    """
    Function for choosing name of trained model from 'models' folder.
    
    -------
    params:
    config - dict (Dotmap) from configuration file with defined parameters values 
             (creates from config_reader function by reading data_config.json)
    is_choice_by_input - boolean. If True - allows to choose and return
                         one of the model names from 'models' folder.
                         If false - return model name from last saved step 

    """


    config.IMG_SIZE = 448
    config.is_show_train_layers = False
    # Creating model & compile model
    model = ModelForTrain(config=config).build_model()
    model.compile(loss=config.loss_compile, 
                  optimizer=optimizers.Adam(learning_rate=config.LR), 
                  metrics=[config.metric_compile])

    path_model_name = choice_model(is_choice_by_input)
    
    if os.path.isfile(path_model_name):
        model.load_weights(path_model_name)
        print(f'{os.path.basename(path_model_name)} are loaded in model')
    
    return model



def load_image(img_path, config, show=True):

    """
    Function for creating image tensor to predict by Keras model.

    -------
    params:

    img_path - path for image 
    config - dict (Dotmap) from configuration file with defined parameters values 
             (creates from config_reader function by reading data_config.json)
    show - boolean. If True - shows image for 5 seconds, False - don't show.
    
    """

    config.IMG_SIZE = 448
    img = image.load_img(img_path, target_size=(config.IMG_SIZE, config.IMG_SIZE))
    img_tensor = image.img_to_array(img)              # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)   # (1, height, width, channels), 
                                                      # add a dimension because the model expects this 
                                                      # shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                # imshow expects values in the range [0, 1]

    if show:
        plt.imshow(img_tensor[0])                           
        plt.axis('off')
        plt.show(block=False)
        plt.pause(5)
        plt.close("all")

    return img_tensor


def decode_image(im_b64_str, config):
    img_bytes = base64.b64decode(im_b64_str.encode('utf-8')) # convert it into bytes
    
    # convert bytes data to PIL Image object
    img = Image.open(io.BytesIO(img_bytes))
    img = img.resize((config.IMG_SIZE, config.IMG_SIZE), Image.Resampling.LANCZOS)

    # PIL image object to numpy array
    img = image.img_to_array(img)
    img_arr = np.expand_dims(img, axis=0)
    img_arr /= 255.
    return img_arr