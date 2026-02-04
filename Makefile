PKG=swasd/src/swasd

lint: 
	flake8 $(PKG)
	isort -c $(PKG)

fix-lint:
	find $(PKG) -name '*.py' | xargs autoflake --in-place --remove-all-unused-imports --remove-unused-variables
	autopep8 --in-place --recursive --aggressive $(PKG)
	isort --atomic $(PKG)

test:
	python -m pytest swasd/tests
