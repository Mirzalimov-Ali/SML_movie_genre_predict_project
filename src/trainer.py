from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsOneClassifier
from imblearn.over_sampling import SMOTE
from collections import Counter
from src.logger import get_logger
import numpy as np

logger = get_logger('training', 'model_training.log')

class Trainer:
    def __init__(self, model, x, y, use_smote=True):
        """
        model: sklearn estimator
        x: features DataFrame
        y: target Series
        use_smote: whether to apply SMOTE for class balancing
        """
        self.model = model
        self.x = x
        self.y = y
        self.use_smote = use_smote
        logger.info(f"Trainer initialized for model: {type(model).__name__}, dataset shape: {x.shape}")

    def train(self):
        try:
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.2, random_state=42)
            logger.info('Dataset split into train/test')

            # SMOTE oversampling
            if self.use_smote:
                smote = SMOTE(random_state=42, k_neighbors=1)
                self.x_train, self.y_train = smote.fit_resample(self.x_train, self.y_train)
                logger.info(f"Applied SMOTE. Train class distribution: {Counter(self.y_train)}")

            # OneVsOne (OVO) wrapper for models which require it
            if type(self.model).__name__ in ['LogisticRegression', 'KNeighborsClassifier']:
                self.clf = OneVsOneClassifier(self.model)
            else:
                self.clf = self.model

            self.clf.fit(self.x_train, self.y_train)
            logger.info(f"Model {type(self.model).__name__} trained")

            # Embedded feature selection
            if hasattr(self.clf, "feature_importances_"):
                importance = self.clf.feature_importances_
                self.selected_features = self.x_train.columns[importance > np.mean(importance)]
                self.clf.fit(self.x_train[self.selected_features], self.y_train)
                logger.info(f"Embedded feature selection applied. {len(self.selected_features)} features selected.")
            else:
                self.selected_features = self.x_train.columns
                logger.info("Feature selection skipped (model does not support feature_importances_)")

            return self

        except Exception as e:
            logger.error(f"Error during training: {str(e)}")

    def evaluate(self):
        try:
            y_pred = self.clf.predict(self.x_test[self.selected_features])
            acc = accuracy_score(self.y_test, y_pred)
            logger.info(f"Prediction done. Accuracy={acc:.3f}")

            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(self.clf, self.x_train[self.selected_features], self.y_train, cv=kf, scoring='f1_macro')

            self.results = {
                "accuracy": acc,
                "kfold_mean": cv_scores.mean(),
                "kfold_std": cv_scores.std(),
                "selected_features": self.selected_features
            }

            return self.results

        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
