import pandas as pd
from src.preprocessing import Preprocessing
import os
from src.logger import get_logger

logger = get_logger('use_fill_missing_values', 'preprocessing.log')

df = pd.read_csv('data/raw/merged_dataset.csv')
logger.info(f'dataset loaded with shape: {df.shape}')

preprocessing = Preprocessing(df, target='Genre')
df = preprocessing.fillMissingValues().getDataset()
logger.info(f'missing values filled successfully')


# Save dataset
output_folder = 'data/filledMissingValues'
os.makedirs('data/filledMissingValues', exist_ok=True)

output_path = os.path.join(output_folder, 'missing_values_filled_dataset.csv')
logger.info(f'file created to path: {output_path}')

df.to_csv(output_path, index=False)

print(df.head(10))