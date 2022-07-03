FROM agrigorev/zoomcamp-model:mlops-3.9.7-slim

WORKDIR /app

COPY batch.py batch.py
COPY requirements.txt requirements.txt

RUN pip install -U pip
RUN pip install -r requirements.txt
RUN mkdir output

CMD python batch.py 2021 4