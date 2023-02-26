from dotmap import DotMap
from glob import glob
import matplotlib.pyplot as plt
import os

from tensorflow import keras
from keras import optimizers
import keras.utils as image

from ImageDataAugmentor.image_data_augmentor import *
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau # LearningRateScheduler
from utils.read_config import config_reader
from utils.model import ModelForTrain



# Callbacks that used for training model
def callbacks(PATH_BEST_MODEL, config):                               

    """
    Add description
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



def save_model(PATH_BEST_MODEL, config, model, step_num):
    
    # load best weights
    model.load_weights(os.path.join(PATH_BEST_MODEL, config.best_model_name))
    
    # Save model & load best iteration after fitting (best_model):
    model.save(os.path.join(PATH_BEST_MODEL, f'model_step_{step_num}.h5'))
    model.save_weights(os.path.join(PATH_BEST_MODEL, f'weights_step_{step_num}.hdf5'))
    print(f'model and weights of {step_num} step traininig are saved in {PATH_BEST_MODEL}')



def eval_model(model, test_generator):
    scores = model.evaluate(test_generator, verbose=1)
    print("Accuracy: %.2f%%" % (scores[1]*100))



def make_predictions(config, test_sub_generator, model, label_map):

    test_sub_generator.reset()
    pred = model.predict(test_sub_generator, steps=len(test_sub_generator),  verbose=1) 
    predictions = np.argmax(pred, axis=-1) #multiple categories
    label_map = dict((v,k) for k,v in label_map.items()) #flip k,v
    predictions = [label_map[k] for k in predictions]
    
    return predictions



def make_tta(config, test_sub_generator, model, label_map):
    
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
    
    submission = pd.DataFrame({'Id':filenames_with_dir, 'Category':predictions}, columns=['Id', 'Category'])
    submission['Id'] = submission['Id'].replace('test_upload/','')
    submission.to_csv('submission.csv', index=False)
    print('Save submit')
    
    return submission


def create_model(config_path):
    
    # Define configs
    config = config_reader(config_path)
    #config.IMG_SIZE = 448
    
    # Creating model & compile model
    model = ModelForTrain(config=config).build_model()
    model.compile(loss=config.loss_compile, 
                optimizer=optimizers.Adam(learning_rate=config.LR), 
                metrics=[config.metric_compile])

    path_model_name = choice_model()

    if os.path.isfile(path_model_name):
        model.load_weights(path_model_name)
    
    print(f'model {os.path.basename(path_model_name)} loaded')

    return model



def load_image(img_path, show=False):

    img = image.load_img(img_path, target_size=(224, 224))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    if show:
        plt.imshow(img_tensor[0])                           
        plt.axis('off')
        plt.show()

    return img_tensor