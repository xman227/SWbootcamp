FROM python:3.10-slim-bullseye


COPY . ./docker

RUN pip install -U flask-cors

EXPOSE 8080

CMD ["python", "./docker/main.py"]
