from pathlib import Path

DATA_PATH = Path('data/')
DATA_PATH.mkdir(parents=True, exist_ok=True)

DATA_CACHE = Path('data/cache_dir/')
DATA_CACHE.mkdir(parents=True, exist_ok=True)

DATA_PATH_SAVE_MODELS = Path('data/models/')
DATA_PATH_SAVE_MODELS.mkdir(parents=True, exist_ok=True)

pd.set_option('display.max_colwidth', 500) 

Path('/data/cache_dir').mkdir(parents=True, exist_ok=True)
Path('/data/cache_dir').mkdir(parents=True, exist_ok=True)
Path('/data/cache_dir').mkdir(parents=True, exist_ok=True)