FROM python:3.11
LABEL authors="sitdi"

WORKDIR /app

COPY poetry.lock pyproject.toml Flask_model.py BrawlerYuraClassifierModel.py .

RUN pip install poetry

RUN POETRY_VIRTUALENVS_CREATE=false poetry install --no-interaction --no-ansi

#COPY wheelhouse /wheelhouse
#RUN pip install --no-index --find-links=/wheelhouse *.whl

CMD ["python","Flask_model.py"]