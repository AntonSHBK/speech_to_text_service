from pathlib import Path


DATA_PATH = Path('app/data/')
DATA_PATH.mkdir(parents=True, exist_ok=True)

DATA_CACHE = Path('app/data/cache_dir/')
DATA_CACHE.mkdir(parents=True, exist_ok=True)

DATA_INPUT = Path('app/data/input/')
DATA_INPUT.mkdir(parents=True, exist_ok=True)

DATA_OUTPUT = Path('app/data/output/')
DATA_OUTPUT.mkdir(parents=True, exist_ok=True)