import pandas as pd
from pathlib import Path
import shutil
from tqdm.auto import tqdm

test_data = pd.read_csv('data/kaggle/kaggle_art_dataset_test.csv')

image_names = test_data['filename']

data_path = Path('/nfs/data/iart/kaggle/img')

result_path = Path('data/kaggle/photos/test')

if not result_path.is_dir():
    result_path.mkdir(parents=True, exist_ok=True)

for image_name in tqdm(image_names):
    shutil.copy(data_path / image_name, result_path / image_name)