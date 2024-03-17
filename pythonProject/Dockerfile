FROM python:3.11
LABEL authors="sitdi"

WORKDIR /app

COPY ../data /app/data
COPY ../pythonProject /app/pythonProject
COPY poetry.lock pyproject.toml .

RUN pip install poetry

RUN POETRY_VIRTUALENVS_CREATE=false poetry install --no-interaction --no-ansi

#COPY wheelhouse /wheelhouse
#RUN pip install --no-index --find-links=/wheelhouse *.whl

CMD ["python","pythonProject/Flask_model.py"]