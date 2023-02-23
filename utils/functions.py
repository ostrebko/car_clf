from dotmap import DotMap
from glob import glob
import matplotlib.pyplot as plt
import os
from skimage import io
import zipfile

from ImageDataAugmentor.image_data_augmentor import *
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau # LearningRateScheduler



def create_paths(config):
    """
    add description params

    """
    paths_dict = dict()

    paths_dict['PATH_DATA'] = os.path.join('..', config.PATH_DATA)

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
    io.imshow(image_RGB)
    io.show()



def extract_data_from_zip(path_to_big_zip, path_data_train, is_true=False):
    """
    add description params

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



def get_label_map(paths_dict):
    
    num_labels = map(os.path.basename, glob(os.path.join(paths_dict.PATH_TO_DATA_TRAIN, '*')))
    label_map_dict = {i : int(i) for i in num_labels}

    return label_map_dict


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
