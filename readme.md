# Car classification


## Summary

<p>This application allows to carry out a multi-class classification of the car model by image. Classification is carried out according to the following 10 classes: "Lada Priora", "Ford Focus", "Lada 2114", "Lada 2107", "Lada Niva", "Lada Kalina", "Lada 2109", "Volkswagen Passat", "Lada 21099". The application presents a notebook for training a model. The basic idea for preparing a notebook is to take a pre-trained model and retrain it to perform the task of classifying a car by image.</p><br>


## Contents

* [Introduction](README.md#Introduction)
* [Project structure](README.md#Project-structure)
* [Loading data](README.md#Loading-data)
* [Instalation](README.md#Instalation)
* [Activation the virtual environment](README.md#Activation-env)
* [Docker](README.md#Docker)
* [Creation exe](README.md#Creation-exe)
* [Inference](README.md#Inference)
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
│   ├── data_2_load.md      ## file with links to training dataset
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




## Other temp text for create readme
<details>
Данные для обучения модели (kaggle competitions download -c sf-dl-car-classification) решения проводятся в соответствующем соревновании на Kaggle (https://www.kaggle.com/c/sf-dl-car-classification). Данные для иных площадок можно скачать из датасета (приведено в файле data_2_load.md).  

Основной ход моего решения заключался в следующем:  
1. Установка и импорт необходимых библиотек, в т.ч. определение основных переменных и создание необходимых папок для сохранения результатов; Прим.: [Решение ошибки в Keras](https://discuss.tensorflow.org/t/using-efficientnetb0-and-save-model-will-result-unable-to-serialize-2-0896919-2-1128857-2-1081853-to-json-unrecognized-type-class-tensorflow-python-framework-ops-eagertensor/12518/9). Для запуска ноутбука в виртуальном окружении: open the Command Palette, and select "Notebook: Select Notebook Kernel" -> далее меняем c global на python path -> выбираем env.  
2. Проведение краткого EDA, в т.ч. анализ имеющихся изображений;
3. Аугментация данных (использовалась библиотека albumentations) и создание соответствующих генераторов (с помощью библиотеки ImageDataAugmentor) для подачи данных в модель при обучении;  
4. Для создания модели использовалась техника Transfer-Learning: как основа загружалась EfficientNetB6 с исключением полносвязных слоев, которые определяют набор вероятностей к каждому классу ImageNet (исключение "головы"). Вместо исключенных слоев достраивались полносвязные слои под нашу задачу.
5. В основе тренировки модели использовалась техника Fine-Tunning: тренировка модели проводилась с постепенным размораживанием весов слоев, доступных для тренировки и состояла из нескольких шагов (step):  

    Step 1 - тренировка весов слоев только для "головы", с неизменными весами EfficientNetB6 (уже после данного этапа точность на тренировочной выборке превышает 50%, на тестовой - превышает 60%). Так как в дальнейшем веса будут переобучаться при разморозке модели, то на данном этапе было выбрано небольшое количество эпох обучения. Точность на тренировочной выборке оказывается хуже, но к 5 эпохе точность тестовой выборки перестает улучшаться, а точность тренировочной выборки растет быстрее.  
    
    Step 2-4 - тренировка с постепенной разморозкой весов слоев EfficientNetB6. Step 2: разморозка 1/2 от всех слоев EfficientNetB6, тренировка 10 эпох; Step 3: разморозка 3/4 от всех слоев EfficientNetB6, тренировка 10 эпох; Step 4: разморозка всех слоев EfficientNetB6, тренировка 10 эпох.      
    Результаты по обучению на шагах 2-4:  
    Наилучшая сходимость тренировочиной и тестовой выборок достигается после Step 2 (разморозка 1/2 от всех слоев EfficientNetB6) на 10 эпохе и составляет чуть больше 90%. На данном этапе можно попробовать большее количество эпох (30-50 эпох) с постепенным (по расписанию или по условию неувеличения val_accuracy) уменьшением Learning Rate. Но так как время на обучение ограничено количеством времени использования GPU и слои будут размораживаться далее, соответственно обученные веса будут еще изменяться, то было решено не работать в этом направлении.  
    На шаге 3 (step 3) размораживаю 3/4 всех слоев и обучаю 10 эпох.
    На шаге 4 (step 4) размораживаю всех слои base_model (всех слоев EfficientNetB6) и обучаю 10 эпох. 
    
    Шаги 5-7 (Step 5, 6, 7: На данных шагах для увеличесния точности обучения модели производится увеличение размера подаваемых изображений в 2 раза (с 224х228 до 448х448 точек). Обучение происходит при всех размороженных слоях, но при этом меняется learning rate: Step 5, LR=1e-5, 8 эпох, Step 6 LR=1e-5, 6 эпох (отключение GPU Colab), Step 7 LR=1e-6, 10 эпох. Примечание: На Step 6 было решено добавить еще 6 эпох без изменения параметров шага 5_1.  
    Важно отметить, что при увеличении картинки в 2 раза, время на обучение возрасло примерно в 3-4 раза и обучение 10 эпох каждого шага растягивается примерно до 6,5 часов. В связи с тем, что на Kaggle есть ограничение на работу одной сессии с GPU (9 часов) и прогнать весь ноутбук и сохранить все результаты можно только исключив данные ограничения. Аналогично есть ограничения на использования GPU и в Google Colab. Поэтому в данной работе я сохранил, только ноутбук и результаты предсказания на валидационной выборке (submission).
    На шаге 7 параметр patience в callback ReduceLROnPlateau , был изменен с 3 на 2 (количество эпох, после которых, если не увеличается точность, то уменьшается learning rate)
    
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