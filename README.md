# ML_module_1

About:
Проект сделан в рамках 1 модуля по предмету machine learning. В данном модуле необходимо было решить задачу Spaceship Titanic. Лучшй результат - 0.81108 (0.80687 в итоге получился, тк не сохранился код с лучшим результатом).
Использовалась модель catBoostClassifier с посиком оптимальных параметров с помощью optuna.
Poetry для замораживания зависимостей и легкой подкачки их.
Rest API при помощи Flask. (файл Flask_model.py).
CLI с помощью fire. (файл CLI_model.py)
Проект докеризирован, смотрите ниже как его задеплоить. (на любой запрос кидает connection refused, пока не разобрался в чем проблема)

fullname & group
Sitdikov Yuriy 972203

How to:

  · CLI:
  
    - клонируйте репозиторий
    
    - перейдите в каталог репозитория
    
    - введите команду poetry install, чтобы установить нужные зависимости
    
    - можно вводить команду "python CLI_model.py train --dataset=/path/to/train/dataset" или "python model.py predict --dataset=/path/to/test/dataset"
    
  · Dockerfile:
  
    - клонируйте репозиторий
    
    - перейдите в каталог с dockerfile
    
    - docker image build -t <your_image_name>
    
    - docker run -p <your_port> <your_image_name>
    
    - можно отправлять запросы "http://localhost:your_port/train?dataset=/path/to/train/dataset" или "http://localhost:your_port/predict?dataset=/path/to/test/dataset"
    
  · Через запуск Flask_model.py
  
    - клонируйте репозиторий
    
    - перейдите в каталог репозитория
    
    - введите команду poetry install, чтобы установить нужные зависимости
    
    - запустите файл Flask_model.py
    
    - можно отправлять запросы на localhost:5000
    

Best params:

  - learning_rate=0.07981609439133353
    
  - depth=5
    
  - iterations=392
    
  - k=13 (число групп в kfold для обучения)
