import pandas as pd
import os
from src.logger import get_logger

logger = get_logger("data_loader", "data_loader.log")

class DataLoader:
    def __init__(self):
        pass

    def load_and_merge(self):
        df_list = []

        files = ["Action_movies.csv", "Family_movies.csv", "Sport_movies.csv", "Horror_movies.csv"] 

        for file in files:
            file_path = os.path.join("data/raw", file)

            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    df_list.append(df)
                    logger.info(f"Loaded file: {file_path}")
                except Exception as e:
                    logger.error(f"Error loading file {file_path}: {e}")
            else:
                logger.warning(f"File not found: {file_path}, skipping.")

        if df_list:
            merged_df = pd.concat(df_list, ignore_index=True)
            merged_df['Id'] = range(1, len(merged_df) + 1)
            logger.info(f"Data merged successfully. Shape: {merged_df.shape}")
            return merged_df
        else:
            logger.error("No data loaded! Check your file paths and filenames.")

    def save_merged(self, df):
        save_path = os.path.join("data/raw", "merged_dataset.csv")
        os.makedirs("data/raw", exist_ok=True)
        df.to_csv(save_path, index=False)
        logger.info(f"Merged data saved to {save_path}")
