.PHONY: install ingest index test serve

install:
	uv venv --python 3.14 .venv
	uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
	uv pip install --python .venv/bin/python -r requirements.txt

ingest:
	.venv/bin/python metadata.py

index:
	.venv/bin/python embeddings.py --batch-size 32

test:
	.venv/bin/python -m pytest

serve:
	.venv/bin/python app.py
