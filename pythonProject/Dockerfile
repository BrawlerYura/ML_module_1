FROM python:3.11
LABEL authors="sitdi"

WORKDIR /app

ADD . /app

COPY wheelhouse /wheelhouse

RUN pip install --no-index --find-links=/wheelhouse *.whl

CMD ["python","Flask_model.py"]