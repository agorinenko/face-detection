ARG LOG_LEVEL

FROM python:3.7

EXPOSE 8005

WORKDIR /app
COPY . .

RUN apt-get update --allow-insecure-repositories
RUN apt-get install ffmpeg libsm6 libxext6 -y --allow-unauthenticated

RUN python -m pip install --upgrade pip
RUN pip install -r requirements.gunicorn.txt

CMD ["gunicorn", "server:app", "-b", "0.0.0.0:8005", "--worker-class", "aiohttp.GunicornWebWorker", "--log-level", "${LOG_LEVEL}"]