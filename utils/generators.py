import os
import pandas as pd
from ImageDataAugmentor.image_data_augmentor import *


class DataGenerators(ImageDataAugmentor):
    """
    Function for creating objects of class ImageDataAugmentor 
    for image data generation with augmentations transformations
    
    Params:
    ----------
    ....
    
    val_split=config.VAL_SPLIT, seed_num=seed_value, albument_transforms=AUGMENTATIONS,
    img_size=config.IMG_SIZE, banch_size_for_gen=config.BATCH_SIZE, DATA_PATH=PATH_DATA_TRAIN,
    sample_submission_df=sample_submission

    subset: 'training' or 'validation' for train_generator or test_generator
    
    """ 

    def __init__(self, config: dict, albument_transforms):
        super().__init__()
        self.config = config
        self.PATH_DATA_TRAIN = os.path.join('..', config.PATH_DATA, config.folder_data_train)
        self.albument_transforms = albument_transforms

        if albument_transforms != None:
            # Creating objects of class ImageDataAugmentor 
            self.train_datagen = ImageDataAugmentor(
                rescale=1./255, validation_split=config.VAL_SPLIT, 
                seed=config.RANDOM_SEED, augment=albument_transforms, preprocess_input=None)
            self.test_datagen = ImageDataAugmentor(rescale=1./255, seed=config.RANDOM_SEED)
        else:
            # Creating objects of class ImageDataAugmentor 
            self.train_datagen = ImageDataAugmentor(
                rescale=1./255, validation_split=config.VAL_SPLIT, seed=config.RANDOM_SEED)
            self.test_datagen = ImageDataAugmentor(rescale=1./255, seed=config.RANDOM_SEED)
        

    def create_generator(self, subset: str):
        
        # Creationg data_generators from methods of objects of class ImageDataAugmentor
        generator_from_dir = self.train_datagen.flow_from_directory(
            os.path.join(self.PATH_DATA_TRAIN, 'train'),  # the directory where the folders with pictures are located
            target_size=(self.config.IMG_SIZE, self.config.IMG_SIZE),
            batch_size=self.config.BATCH_SIZE,
            class_mode='categorical',
            shuffle=True, 
            subset=subset # set as training data
            )

        return generator_from_dir


    def create_test_sub_generator(self, sample_submission_df: pd.DataFrame, generator: ImageDataAugmentor):
        
        test_sub_generator = generator.flow_from_dataframe( 
            dataframe=sample_submission_df,
            directory=os.path.join(self.PATH_DATA_TRAIN, 'test_upload'),
            x_col="Id",
            y_col=None,
            shuffle=False,
            class_mode=None,
            target_size=(self.config.IMG_SIZE, self.config.IMG_SIZE),
            batch_size=self.config.BATCH_SIZE,
            )
    
        return test_sub_generator