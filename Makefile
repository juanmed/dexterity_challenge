.PHONY: build-cpp test run-dev

build-cpp:
	mkdir -p src/dexterity/cpp/build
	cd src/dexterity/cpp/build && cmake .. && make

test:
	uv run pytest tests/ -v --cov=src/dexterity --cov-report=term-missing

run-dev:
	uv run python scripts/run_dev.py $(N)
