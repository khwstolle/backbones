
help:
	@echo "install - install the package"
	@echo "check - run linters and formatters"
	@echo "test - run tests"
	@echo "benchmark - run benchmarks"
	@echo "coverage - run coverage"
	@echo "build - build the package"
	@echo "dist - build and upload the package"
	@echo "clean - clean the project"

clean:
	rm -rf build dist *.egg-info .pytest_cache .tox .coverage .hypothesis .mypy_cache .mypy .ruff .ruff_cache .pytest_cache .pytest .benchmarks .benchmarks_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	
install:
	pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation ./

check:
	ruff check --fix .

test: check
	python -m pytest -s -v -n auto --dist=loadfile --junitxml=tests.xml --no-cov --benchmark-disable
	
benchmark:
	python -m pytest -s -v -n 0 --no-cov benchmarks

coverage:
	python -m pytest --cov=sources --cov-report=html --cov-report=xml --benchmark-disable

build: test
	python -m build --wheel

dist: build
	python -m twine check dist/*
	python -m twine upload dist/*

.PHONY: help install check test benchmark coverage dict build clean