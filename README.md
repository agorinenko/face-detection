# Реализация веб-демо с моделью детектирования

## Install dev requirements

```shell script
pip install -r requirements.dev.txt
```

## Start dev server

Вызвать из корневого каталога проекта

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

## Start in docker

Create network

```shell script
docker network create
```