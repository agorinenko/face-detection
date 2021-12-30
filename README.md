# Implementation of a web demo with an image detection model

Link to jupyter notebook

https://colab.research.google.com/drive/1J73R_8MC9OAqes4S2WyrpS9BLoP-2cE_?usp=sharing

## Start app in docker

1. Install docker and docker-compose https://docs.docker.com/compose/install/
2. Execute from the root directory of the project

```shell script
docker-compose up -d --build
```

Result on my PC

```shell
[+] Building 227.1s (22/22) FINISHED 
[+] Running 2/2
 ⠿ Container face_detection_app_1    Started                                                                                                                                                                                                                                                                                                                                                                                     3.2s
 ⠿ Container face_detection_nginx_1  Started  
```

3. Open http://127.0.0.1/
4. Profit!

## Install dev requirements

```shell script
pip install -r requirements.dev.txt
```

## Start dev server

Execute from the root directory of the project

```shell script
python -m api runserver api --root api --verbose
```

## Run tests from console

```shell script
pytest -ra
```

## PyCharm :: Run/Debug configuration

1. Add python configuration
2. Module name: api
3. Parameters: runserver api --root api --verbose
4. Working directory - укажите актуальные корневой каталог проекта

## Environment variables

See ".env.example" file

```shell
cp .env.example .env
```