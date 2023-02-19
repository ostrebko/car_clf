from tensorflow import keras
from keras.models import Model
from keras.applications.efficientnet import EfficientNetB6 as BaseTrainedModel
from keras.layers import GlobalAveragePooling2D, Dense 
from keras.layers import BatchNormalization, Dropout


class ModelForTrain(Model):
    """
    Class creates model e.g. EfficientNetB6, inheriting class from tensorflow.keras.models.
    
    Parameters:
    ----------
    
    """    

    def __init__(self, config: dict):
        
        super().__init__()
        
        self.is_show_summary = config.is_show_summary
        # ------- parameters ------------
        #self.config = config
        
        # -------- model layers ----------------
        self.base_model = BaseTrainedModel(weights=config.weights, 
                                           include_top=config.include_top, 
                                           input_shape=(config.IMG_SIZE, config.IMG_SIZE, 
                                                        config.IMG_CHANNELS)
                                           )

        self.base_model.trainable = config.trainable
        
        # Сhoose layers which weights will train and freeze  
        if config.train_all_base_layers==False:
            # Fine-tune from this layer onwards
            fine_tune_at = int(len(self.base_model.layers)//config.f_tune_coef)
            # Freeze all the layers before the `fine_tune_at` layer
            for layer in self.base_model.layers[:fine_tune_at]:
                layer.trainable = False
        

        # Creation new head:
        x = self.base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(units=config.dense_units, activation=config.dense_activation)(x)
        x = BatchNormalization()(x)
        x = Dropout(config.dropout_ratio)(x)
        
        # Logistic layer -- 10 classes
        self.predictions = Dense(config.CLASS_NUM, activation=config.output_activation)(x)


    def build_model(self):
        """
        Method of creation model 
        """
        model = Model(
            inputs=self.base_model.input,
            outputs=self.predictions,
            name="Custom_model"
        ) 
        if self.is_show_summary:
            model.summary() # shows model summary
            
            # numbers of layers and training variables
            print(f'Number of model layers: {len(model.layers)}')
            print(f'Number of trainable_variables in model: {len(model.trainable_variables)}')
        
        else:
            print(f'Number of model layers: {len(model.layers)}')
            print(f'Number of trainable_variables in model: {len(model.trainable_variables)}')
        
        return model