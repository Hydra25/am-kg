# Shortcuts:  make test | make lint | make fix | make app
PY ?= python

test:
	$(PY) -m pytest

lint:
	ruff check .

fix:
	ruff check --fix .

app:
	$(PY) -m streamlit run app/streamlit_app.py
