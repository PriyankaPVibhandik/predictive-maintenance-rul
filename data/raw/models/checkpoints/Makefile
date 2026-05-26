.PHONY: install preprocess train anomaly api dashboard test mlflow

install:
	pip install -r requirements.txt

preprocess:
	python -m src.data.preprocess

train:
	python -m src.models.train

anomaly:
	python -m src.models.anomaly

api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	streamlit run src/dashboard/app.py

mlflow:
	mlflow ui --port 5000

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

all: preprocess train anomaly
