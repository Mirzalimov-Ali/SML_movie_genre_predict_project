import pandas as pd
from src.preprocessing import Preprocessing
import os
from src.logger import get_logger

logger = get_logger('use_preprocessing', 'preprocessing.log')

df = pd.read_csv('data/engineered/engineered_dataset.csv')

preprocessing = Preprocessing(df, target='Genre')
df = preprocessing.encode().scale().getDataset(update_columns=False)
df = preprocessing.logTransformation(df)


# Save dataset
output_folder = 'data/preprocessed'
os.makedirs('data/preprocessed', exist_ok=True)

output_path = os.path.join(output_folder, 'preprocessed_dataset.csv')
logger.info(f'file created to path: {output_path}')

df.to_csv(output_path, index=False)

print(df.head(10))