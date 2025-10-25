from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator
import numpy as np
import pandas as pd
from src.logger import get_logger

logger = get_logger('preprocessing', 'preprocessing.log')

class Preprocessing(BaseEstimator):
    def __init__(self, df, target):
        self.df = df.copy()
        self.target = target

        x = self.df.drop(columns=[self.target])
        self.num_col = x.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_col = x.select_dtypes(exclude=[np.number]).columns.tolist() 

        self.numerical_pipeline = None
        self.categorical_pipeline = None
        self.preprocessor = None
        
        logger.info(f'Preprocessing initialized successfully! Found {len(self.num_col)} numerical and {len(self.cat_col)} categorical features.')

    def fillMissingValues(self):
        try:
            self.df = self.df.dropna(subset=['Description']).reset_index(drop=True)

            self.numerical_pipeline = Pipeline([('imputer', SimpleImputer(strategy='mean'))])
            self.categorical_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent'))])

            logger.info('Pipelines for filling missing values created successfully')
            return self
        
        except Exception as e:
            logger.error(f'Error while creating pipelines for missing values!: {e}')

    def encode(self):
        try:
            self.df[self.cat_col] = self.df[self.cat_col].astype(str)
            
            self.categorical_pipeline = Pipeline([
                ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
            ])
            logger.info('Categorical features successfully encoded using OrdinalEncoder()')

            if self.df[self.target].dtype == object or str(self.df[self.target].dtype).startswith('category'):
                self.df[self.target] = self.df[self.target].astype('category').cat.codes
                logger.info(f'Target column {self.target} encoded using cat.codes')

            return self

        except Exception as e:
            logger.error(f'Error while encoding categorical features: {e}')


    def scale(self):
        try:
            self.numerical_pipeline = Pipeline([('scaler', MinMaxScaler())])
            logger.info('Numerical scaling with MinMaxScaler() configured successfully.')

            return self

        except Exception as e:
            logger.error(f'Error while configuring numerical scaling: {e}')


    def logTransformation(self, x):
        try:
            num_cols = x.select_dtypes(include=[np.number]).columns
            skewness = x[num_cols].skew()
            features_log = skewness[skewness >= 0.5].index.tolist()

            for col in features_log:
                if (x[col] > 0).all():
                    x[col] = np.log1p(x[col])

            logger.info(f'Log transformation applied to: {features_log}')
            return x

        except Exception as e:
            logger.error(f'Error while doing logTransformation: {e}')
    
    def getDataset(self, update_columns=True):
        try:
            x = self.df.drop(columns=[self.target])  
            
            if update_columns:
                x = self.df.drop(columns=[self.target])
                self.num_col = x.select_dtypes(include=[np.number]).columns.tolist()
                self.cat_col = x.select_dtypes(exclude=[np.number]).columns.tolist()

            self.preprocessor = ColumnTransformer([
                ('numerical', self.numerical_pipeline, self.num_col),
                ('categorical', self.categorical_pipeline, self.cat_col)
            ])

            y = self.df[self.target]

            x_transformed = pd.DataFrame(
                self.preprocessor.fit_transform(x),
                columns=self.num_col + self.cat_col
            )

            df = pd.concat([x_transformed, y.reset_index(drop=True)], axis=1)

            logger.info(f'Final dataset created with shape: {df.shape}')
            return df

        except Exception as e:
            logger.error(f'Error in getDataset: {e}')
