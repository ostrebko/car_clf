# Car classification


## Summary

<p>This application allows to carry out a multi-class classification of the car model by image. Classification is carried out according to the following 10 classes: "Lada Priora", "Ford Focus", "Lada 2114", "Lada 2107", "Lada Niva", "Lada Kalina", "Lada 2109", "Volkswagen Passat", "Lada 21099". The application presents a notebook for training a model. The basic idea for preparing a notebook is to take a pre-trained model and retrain it to perform the task of classifying a car by image.</p><br>


## Contents

* [Introduction](README.md#Introduction)
* [Project structure](README.md#Project-structure)
* [Instalation](README.md#Instalation)
* [Activation the virtual environment](README.md#Activation-env)
* [Loading data](README.md#Loading-data)
* [Training model](README.md#Training-model)
* [Inference](README.md#Inference)
* [Docker](README.md#Docker)
* [Flask](README.md#Flask)
* [Conclusions](README.md#Conclusions) <br><br>


## Introduction
<p>
The classification of cars by model is an actual problem and finds application in business and in ensuring safety. The solution of this problem with using machine learning models can be carried out if a lot of time is spent on classification, for example, when processing databases, or the object is in the field of view for a short time, which is not enough to identify the object. As an example, the recognizing of the number and model of the car in any application or during the operation of the security system.<br>

<p align="center">
  <img src="https://volokno.kz/wa-data/public/shop/products/09/10/1009/images/2358/2358.970.jpg" height="200" title="Work with databases">
  <img src="https://news-ru.gismeteo.st/2020/07/shutterstock_443707396-640x427.jpg" height="200" title="Traffic regulations">
</p>
<p align="center">
  <img src="https://avatars.mds.yandex.net/i?id=a7700bda361df26e6eb36d4c9c4a09cc-4080622-images-thumbs&ref=rim&n=33&w=281&h=188" height="200" title="Check payment for parking">
</p>

The task of developing an application includes task of training a machine learning model. To train the model, we will use photos from ads for the sale of cars that are in the public domain.

<p align="center">
  <img src="data/test_imgs_for_pred/352.jpg" height="120" title="lada_niva">
  <img src="data/test_imgs_for_pred/667.jpg" height="120" title="lada_kalina">
  <img src="data/test_imgs_for_pred/3258.jpg" height="120" title="lada_2107">
</p>

<p align="center">
  <img src="data/test_imgs_for_pred/65444.jpg" height="120" title="ford_focus">
  <img src="data/test_imgs_for_pred/4694.jpg" height="120" title="volkswagen_passat">
</p>

The analysis of similar images can lead to an error in human recognition. For example, some car models are generally similar and have slight differences. 

<p align="center">
  <img src="data/test_imgs_for_pred/3201.jpg" height="120" title="lada_2110">
  <img src="data/test_imgs_for_pred/4052.jpg" height="120" title="lada_priora">
</p>
<p align="center">
  <img src="data/test_imgs_for_pred/8846.jpg" height="120" title="2109">
  <img src="data/test_imgs_for_pred/295500.jpg" height="120" title="21099">
</p>

 This task was solved within the framework of [Kaggle competition](https://www.kaggle.com/competitions/sf-dl-car-classification).
</p><br>


## Project structure
<details>
<summary>Display project structure </summary> <br>

```Python
car_clf  
├── .gitignore  
├── .venv  
│   └── ...  
├── config  
│   └── data_config.json    ## congiguration file  
├── data  
│   ├── best_models         ## save best model during train
│   ├── inputs_for_train    ## folder for data
│   │   ├── test_upload     ## folder for test data
│   │   ├── train           ## folder for train data
│   │   ├── sample-submission.csv  ## ex file for kaggle submission
│   │   └── train.csv  
│   ├── outputs_from_train  ## folder for saved graph
│   ├── test_imgs_for_pred  ## folder with few samples in test_upload
│   └── sf-dl-car-classification.zip ## uploaded zip train dataset 
├── models                  ## folder for trained model
│   ├── weights_step_1.hdf5  
│   ├── ...  
│   └── weights_step_7.hdf5  
├── notebooks               ## notebook for create train models
│   ├── 01_notebook_train_model.ipynb  
│   ├── 02_colab_notebook_train_model.ipynb  ## colab notebook
│   └── 03_car-clf-nn-2021_OLD_ver.ipynb  
├── utils  
│   ├── __ init __.py  
│   ├── functions.py  
│   ├── functions_with_keras.py  
│   ├── generators.py  
│   ├── model.py  
│   ├── predictions.py  
│   └── read_config.py  
├── readme.md  
└── requirements.txt
```
</details>  <br>


## Instalation
<details>

<summary> Display how to install app </summary> <br>

<p> This section provides a sequence of steps for installing and launching the application. <br>

```Python
# 1. Clone repository
git clone https://github.com/ostrebko/car_clf.git

# 2. Go to the new directory:
cd car_clf

# 3. Activate the virtual environment in which you plan to launch the application (we will use VsCode)

# 4. Install requirements:
pip install -r requirements.txt

# 5. Create predicts of detection blastospores with main.py or create & run main.exe (in windows).
python main.py
```
</details>  <br>


## Activation env
<details>

<p> The description of how to activate the virtual environment was taken from <a href="https://kayumov.ru/536/">Ruslan Kayumov</a>.<br>

<summary> Type in the console: </summary> <br>

```Python
# Steps to activate the virtual environment in which you plan to launch the application in VsCode:
# 1. Run VS Code as an administrator, go to the project directory in PowerShell, execute the code below, the env folder containing the virtual environment files will appear
python -m venv .venv

# or you may tap -> Ctrl+Shift+P , then press -> Python: Select Interpreter (we use venv), choose 'Python 3.хх.хх ... Global' for create the virtual environment with GUI of VS Code.

# 2. To change the policy, in PowerShell type
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 3. Enter the environment folder (env), run the command
.venv/Scripts/Activate.ps1

# 4a. An environment marker (env) will appear at the beginning of the line in PowerShell, but VS Code may still not know anything about it. Press Ctrl+Shift+P, type Python: Select Interpreter
# Specify the desired path to python.exe in the env environment folder, this will be displayed at the bottom of the status bar. Now you can install modules only for a specific project.

# 4b. For VSCode, your Jupyter kernel is not necessarily using the same python interpreter you're using at the command line but if you have special libs you may need to using your notebook in created virtual environment.
# For using your notebook in created virtual environment install ipykernel:
pip install ipykernel
# then tap Ctrl+Shift+P to open the Command Palette, and select "Notebook: Select Notebook Kernel" ->
# -> Select another kernel -> Python Environments -> choose the interpreter you're using at the terminal (we create virtual environment with name: .venv)

# 5. If you need to exit, then execute deactivate in PowerShell, and return to global in the interpreter selection.
```
</details>


## Loading data
<details>
<p>Before training the model, it's necessary to download training dataset. **Attention**, it requires **1,66 Gb** of free disk spaces.</p>
<p>If you want to train the model in local machine, go to the [**Kaggle web page**](https://www.kaggle.com/competitions/sf-dl-car-classification/data/) in your browser and tap **Download All**, the file *'sf-dl-car-classification.zip'* will start downloading. Then move downloaded *'sf-dl-car-classification.zip'* in 'data' folder of the cloned project.</p>
<p>If you use notebook **'02_colab_notebook_train_model.ipynb'** for train model with Google Colab, you don't need to download data to local machine. Notebook consist cells with code to download *'sf-dl-car-classification.zip'* into cloned project in Goole Colab environment.</p>
<p>You also may to use Kaggle API to download -> *'kaggle competitions download -c sf-dl-car-classification'* 
(see. [Kaggle API in github](https://github.com/Kaggle/kaggle-api) ).</p>
<p>To unzip *'sf-dl-car-classification.zip'* into cloned project use the corresponding notebook cells.</p>
</details>


## Training model
<details>
<p>Basic steps of model preparation:  

1. Installing and importing the necessary libraries, functions and classes, fixing seed values, creating the necessary folders for data and saving results and unpacking the 'sf-dl-car-classification' archive.zip' (if not done earlier);  
Note: [Solving a possible error in Keras](https://discuss.tensorflow.org/t/using-efficientnetb0-and-save-model-will-result-unable-to-serialize-2-0896919-2-1128857-2-1081853-to-json-unrecognized-type-class-tensorflow-python-framework-ops-eagertensor/12518/9)  
2. Conducting a brief EDA, including analysis of available images;
3. Сreating a data augmentation object (using the **Albumentations** library) and creating data generators (using the **ImageDataAugmentor** library) to feed data in batches to the model during training;
4. The **Transfer-Learning** technique was used to create the model. As a basis, **EfficientNetB6** was loaded with the exception of fully connected layers (excluding the "head"). Instead of the excluded layers, fully connected layers were completed for our task. To create a model for training, the **ModelForTrain** class was written with a calling *build_model* method;  
5. The model training was based on the **Fine-Tuning** technique: the model training was carried out with gradual defrosting of the model layers and consisted of several steps (step):  
    **Step 1** - training of layer weights only for the "head", with constant EfficientNetB6 weights (after this step, the accuracy on the training sample exceeds 50%, on the test sample exceeds 60%). Since in the future the weights will be retrained when the model is defrosted, a small number of training epochs were selected at this stage. Note: The accuracy on the training sample turns out to be worse than the accuracy on the test sample, but by the 5th epoch, the accuracy of the test sample ceases to improve, and the accuracy of the training sample grows faster (see. Train history of accuracy and loss in Pic.1).<br>

    <p align="center">
      <img src="data/outputs_from_train/step_1_acc_train.png" height="240" title="history_acc_train step_1">
      <img src="data/outputs_from_train/step_1_loss_train.png" height="240" title="history_loss_train step_1">
    </p> 

    **Step 2-4** - training with gradual defrosting of body weights (i.e. layers of EfficientNetB6). Step 2: defrost 1/2 from all layers EfficientNetB6, training 10 epochs; Step 3: defrost 3/4 from all layers EfficientNetB6, training 10 epochs; Step 4: defrost all layers EfficientNetB6, training 10 epochs.<br> 
    Learning outcomes in steps 2-4:<br>
    The best convergence of the training and test samples is achieved after **Step 2** (defrosting 1/2 of all the layers of EfficientNetB6) at the 10th epoch and is slightly more than 90%. At this stage, you can try a larger number of epochs (30-50 epochs) with a gradual (according to the schedule or according to the condition of non-exaggeration of val_accuracy) decrease in the Learning Rate. But since in Colab the training time is limited by the amount of GPU usage time and the layers will be unfrozen further, respectively, the trained weights will still change, it was decided not to work in this direction.<br>

    <p align="center">
      <img src="data/outputs_from_train/step_2_acc_train.png" height="240" title="history_acc_train step_2">
      <img src="data/outputs_from_train/step_2_loss_train.png" height="240" title="history_loss_train step_2">
    </p> 

    At **step 3**, 3/4 of all layers was defrosted and 10 epochs trained.<br>

    <p align="center">
      <img src="data/outputs_from_train/step_3_acc_train.png" height="240" title="history_acc_train step_3">
      <img src="data/outputs_from_train/step_3_loss_train.png" height="240" title="history_loss_train step_3">
    </p> 

    At **step 4**, all base_model layers (all EfficientNetB6 layers) was defrosted and 10 epochs trained.<br>

    <p align="center">
      <img src="data/outputs_from_train/step_4_acc_train.png" height="240" title="history_acc_train step_4">
      <img src="data/outputs_from_train/step_4_loss_train.png" height="240" title="history_loss_train step_4">
    </p> 

    After **Step 3** the accuracy and loss of train and test dataset are diverge, but accuracy on test dataset has a better value than in the previous **Step 2** (see Pic.3). So we try to defrost all layers and train **Step 4**. After **Step 4** it can be seen that the accuracy and loss of train and test dataset are diverge less. The test accuracy continues to grow, and loss continues to continues to decrease. So it time to try **Step 5** to get better training results.<br>

    **Steps 5, 6, 7**: At these steps, in order to increase the accuracy of training the model, the size of the submitted images is increased by 2 times (from 224x228 to 448x448 dots). Learning occurs with all unfrozen layers, but the learning rate changes: **Step 5**: LR=1e-5, 8 epochs; **Step 6**: LR=1e-5, 6 epochs (cause Colab disabling GPU); **Step 7** LR=1e-6, 10 epochs. Note: On **Step 6**, it was decided to add 6 epochs without changing the parameters of **Step 5**.<br>
    It is important to note that when the image is enlarged by 2 times, the training time has increased by about 3-4 times and the training of 10 epochs of each step stretches to about 6.5 hours. Due to the fact that Google Colab has a limit on the operation of one session with the GPU, **Step 5** and **Step 6** were separated. If you can to train 20 epoch without stopping train, change: config.EPOCHS = 20 and skip **Step 6**.<br>
    At **Step 7**, the only *patience* parameter in the callback *ReduceLROnPlateau* was changed from 3 to 2.<br>

</details>


## Inference
<details>
<summary>General description </summary> <br>
<p>The term inference in this project means proving multi-classification of car images with trained model. The application gets to the entrance image, converts image to an array for feeding to the model input and makes a prediction with trained model.</p>  

<p>To carry out an inference perform in the terminal:
```Python
python main.py
```
Then follow the prompts and choose the mode of operation of the program: demonstration mode or manual input of the image path.</p>
</details>


## Docker
<details>

<summary> Display how to create and run docker image  </summary> <br>

```Python
# 1. Create a new image (its size is approximately 3.5 Gb)
docker build -t car_clf .

# 2. Run image in container.
docker run --rm -v $PWD/data/test_imgs_for_pred/:/data/test_imgs_for_pred  --name car_clf car_clf

# 3. The created container will be automatically deleted 
# after executing a sequence of commands from the Dockerfile.  
# Delete the container and image after usage
docker rmi car_clf
```
</details>


## Flask
<details>
add description
.....
</details>


## Other temp text for create readme
<details>

 
    
    
6. Далее для возможного улучшения предсказания качества модели на валидационной выборке использовалась техника Test Time Augmentations, которая основывается на небольших изменениях данных валидационной выборки (аугментация валидационной выборки) и усреднении полученных предсказаний (небольшие изменения могут помочь модели правильно предсказать класс изображения).

Для сокращения написания кода на каждом шаге были написаны функции и классы в т.ч.:  
- запись используемых параметров в data_config.json и его импорт в ноутбук;
- функция создания генераторов данных;
- класс определения архитектуры модели модели;
- функция сборки листа callbacks при обучении модели; 
- функции сохранения и вывода на экран accuracy и loss по эпохам после обучения модели для анализа качества обучения модели; 
- функция сохранения модели в отдельную папку проекта;
- функция выполнения предсказания класса фотографии (инференса модели).  


Результаты предсказания модели на валидационной выборке представлены в файле submission.csv.   
Так как при обучении модели файлы получаются большого объема (до 450 Мб), то их веса выложены в облачном хранилизе: https://drive.google.com/drive/folders/1myedVEqymkIYCOzOj18ChFHfSvswdRv1?usp=sharing. Для проведения обучения в ноутбуке или инференса их необходимо поместить в папку 'models'.   


Что еще можно сделать для улучшения модели/доработки проекта:
1. Попробовать другие архитектуры сетей из SOTA на ImageNet позднее B6, дающие бОльшую точность, например ImageNetB7 или более точные SOTA.  
2. Поэкспериментировать с архитектурой «головы» (например, добавить еще 1-2 полносвязных слоев).  
3. Попробовать больше эпох на 5 этапе обучения (увеличить до 30 эпох с callback ReduceLROnPlateau с параметрами monitor='val_accuracy', factor=0.2-0.5, patience=3-5).  
4. Использовать внешние датасеты для дообучения модели.  
5. Обернуть модель в сервис на Flask (чтобы на практике отследить особенности внедрения DL-моделей в продакшн).  

</details>