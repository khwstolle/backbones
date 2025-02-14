.PHONY: install test build upload clean check

install:
	pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation ./

check:
	ruff check --fix sources tests setup.py

test:
	python -m pytest --benchmark-disable -s -v tests


build: test
	python -m build


upload: build
	twine upload dist/*


clean:
	rm -rf build dist *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
