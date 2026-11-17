FROM python:3.12-slim

RUN pip install --no-cache-dir mlflow==3.16.1 psycopg[binary]==3.2.6

EXPOSE 5000
