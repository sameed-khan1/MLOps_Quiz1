# Makefile for MLOps Quiz Automation

PYTHON = C:\Users\samee\AppData\Local\Programs\Python\Python312\python.exe
PIP = C:\Users\samee\AppData\Local\Programs\Python\Python312\Scripts\pip.exe

.PHONY: all preprocess train evaluate clean help install

help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make preprocess - Run data preprocessing"
	@echo "  make train      - Train the logistic regression model"
	@echo "  make evaluate   - Evaluate the model"
	@echo "  make all        - Run the full pipeline (preprocess, train, evaluate)"
	@echo "  make clean      - Remove generated files"

install:
	$(PIP) install -r requirements.txt

preprocess:
	$(PYTHON) src/preprocess.py

train:
	$(PYTHON) src/train.py

evaluate:
	$(PYTHON) src/evaluate.py

all: install preprocess train evaluate

clean:
	rm -rf data/processed/* models/* results/*
