Задача состоит в построении модели классификации изображений автомобилей по их фотографиям. В выборках присутствуют 10 классов: 'Приора', 'Ford Focus', 'Самара', 'ВАЗ-2110', 'Жигули', 'Нива', 'Калина', 'ВАЗ-2109', 'Volkswagen Passat', 'ВАЗ-21099'.  

Основная идея решения: взять предобученую модель и дообучить под задачу классификации автомобиля по изображению (в моем решении выбрана сеть EfficientNetB6, так как она относится к SOTA на ImageNet, имеет хорошее качество и относительно не большая).  

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


Cтруктура проекта:  
car_clf  
├── .gitignore  
├── .venv  
│   └── ...  
├── config  
│   └── data_config.json  
├── data  
│   ├── best_models  
│   ├── inputs_for_train    
│   │   ├── test_upload  
│   │   ├── train  
│   │   ├── sample-submission.csv  
│   │   ├── test_imgs_for_pred
│   │   └── train.csv  
│   ├── outputs_from_train  
│   ├── data_2_load.md  
│   └── sf-dl-car-classification.zip  
├── models  
│   ├── weights_step_1.hdf5  
│   ├── ...  
│   └── weights_step_7.hdf5  
├── notebooks  
│   ├── 01_notebook_train_model.ipynb  
│   ├── 02_colab_notebook_train_model.ipynb  
│   └── 03_car-clf-nn-2021 OLD_ver.ipynb  
├── utils  
│   ├── __ init __.py  
│   ├── functions.py  
│   ├── functions_with_keras.py  
│   ├── generators.py  
│   ├── model.py  
│   ├── predictions.py  
│   └── read_config.py  
├── readme.md  
├── requirements.txt  
└── submission.csv  


Результаты предсказания модели на валидационной выборке представлены в файле submission.csv.   
Так как при обучении модели файлы получаются большого объема (до 450 Мб), то их веса выложены в облачном хранилизе: https://drive.google.com/drive/folders/1myedVEqymkIYCOzOj18ChFHfSvswdRv1?usp=sharing. Для проведения обучения в ноутбуке или инференса их необходимо поместить в папку 'models'.   


Что еще можно сделать для улучшения модели/доработки проекта:
1. Попробовать другие архитектуры сетей из SOTA на ImageNet позднее B6, дающие бОльшую точность, например ImageNetB7 или более точные SOTA.  
2. Поэкспериментировать с архитектурой «головы» (например, добавить еще 1-2 полносвязных слоев).  
3. Попробовать больше эпох на 5 этапе обучения (увеличить до 30 эпох с callback ReduceLROnPlateau с параметрами monitor='val_accuracy', factor=0.2-0.5, patience=3-5).  
4. Использовать внешние датасеты для дообучения модели.  
5. Обернуть модель в сервис на Flask (чтобы на практике отследить особенности внедрения DL-моделей в продакшн).  
