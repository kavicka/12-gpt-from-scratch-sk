.PHONY: sync data tokenizer baselines train generate plot lint all

STEPS ?= 6000

sync:
	uv sync

data:
	uv run python data.py

tokenizer:
	uv run python bpe.py && uv run python prepare.py

baselines:
	uv run python baselines.py

train:
	uv run python train.py --size base --steps $(STEPS)
	uv run python train.py --size small --steps $(STEPS)
	uv run python train.py --size tiny --steps $(STEPS)

generate:
	uv run python generate.py

plot:
	uv run python plot.py

lint:
	uv run ruff check . && uv run ruff format --check .

all: sync data tokenizer baselines train generate plot lint
