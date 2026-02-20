.PHONY: test fmt

test:
	pytest -q

fmt:
	python -m compileall .
