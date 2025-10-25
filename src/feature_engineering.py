from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import re
from src.logger import get_logger
from joblib import dump

logger = get_logger("feature_engineering", "feature_engineering.log")

class FeatureEngineering:
    def __init__(self, df, text_columns=None, max_features=500, save_tfidf_path=None):
        self.df = df.copy()
        self.text_columns = text_columns if text_columns else ['Description']
        self.max_features = max_features
        self.tfidf = TfidfVectorizer(max_features=self.max_features)
        self.save_tfidf_path = save_tfidf_path
        logger.info(f"FeatureEngineering initialized with dataset shape: {self.df.shape}")

    def create_tfidf_features(self):
        try:
            self.df['Description_length'] = self.df['Description'].fillna('').apply(len)

            def to_minutes(x):
                h = re.search(r'(\d+)h', str(x))
                m = re.search(r'(\d+)m', str(x))
                return (int(h.group(1)) * 60 if h else 0) + (int(m.group(1)) if m else 0)

            self.df['Duration_min'] = self.df['Duration'].apply(to_minutes)
            self.df['Duration_category'] = self.df['Duration_min'].apply(
                lambda x: 'short' if x < 90 else ('medium' if x < 120 else 'long')
            )

            # ----- TF-IDF -----
            self.df['text_combined'] = self.df[self.text_columns].fillna('').agg(' '.join, axis=1)
            X_tfidf = self.tfidf.fit_transform(self.df['text_combined'])
            tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=[f"tfidf_{i}" for i in range(X_tfidf.shape[1])])
            tfidf_df.index = self.df.index
            self.df = pd.concat([self.df, tfidf_df], axis=1)

            if self.save_tfidf_path:
                dump(self.tfidf, self.save_tfidf_path)
                logger.info(f"TF-IDF vectorizer saved to {self.save_tfidf_path}")

            logger.info(f"features created successfully! Dataset shape: {self.df.shape}")
            return self

        except Exception as e:
            logger.error(f"Error in create_tfidf_features: {e}")

    def getDataset(self):
        return self.df
