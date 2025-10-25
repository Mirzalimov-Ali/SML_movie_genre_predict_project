import pandas as pd
from src.logger import get_logger
from src.feature_engineering import FeatureEngineering  
import os

logger = get_logger('use_feature_engineering', 'feature_engineering.log')

# Datasetni yuklash
df = pd.read_csv('data/filledMissingValues/missing_values_filled_dataset.csv')
logger.info(f'dataset loaded with shape: {df.shape}')

# Feature engineering
fe = FeatureEngineering(df, text_columns=['Title', 'Description'], max_features=500, save_tfidf_path='model/best/tfidf_vectorizer.joblib')
df = fe.create_tfidf_features().getDataset()
logger.info(f'new TF-IDF features added')

# Datasetni saqlash
output_folder = 'data/engineered'
os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, 'engineered_dataset.csv')

df.to_csv(output_path, index=False)
logger.info(f'file saved to path: {output_path}')

print(df.head(10))
