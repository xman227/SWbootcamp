FROM tensorflow/tensorflow:2.5.0

RUN pip install --upgrade pip==21.3.1

COPY . ./docker

RUN pip install -r ./docker/requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "./docker/app.py"]