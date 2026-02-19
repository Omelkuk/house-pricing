import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Класс для генерации признаков (Feature Engineering)
class HouseFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # 1. Общая площадь 
        X['TotalSF'] = X['TotalBsmtSF'] + X['1stFlrSF'] + X['2ndFlrSF']
        
        # 2. Суммарное количество ванных комнат
        X['TotalBath'] = (X['FullBath'] + (0.5 * X['HalfBath']) +
                          X['BsmtFullBath'] + (0.5 * X['BsmtHalfBath']))
        
        # 3. Возраст дома и время с последнего ремонта на момент продажи
        X['HouseAge'] = X['YrSold'] - X['YearBuilt']
        X['RemodAge'] = X['YrSold'] - X['YearRemodAdd']
        
        # 4. Суммарная площадь веранд и террас
        X['TotalPorchSF'] = (X['OpenPorchSF'] + X['EnclosedPorch'] + 
                             X['3SsnPorch'] + X['ScreenPorch'] + X['WoodDeckSF'])
        return X


class PreprocessorBuilder:
    @staticmethod
    def build(X: pd.DataFrame) -> ColumnTransformer:
        # Список колонок, где пропуск (NaN) — это отсутствие объекта
        big_na_cols = [
            'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 
            'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
            'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'
        ]

        # Автоматическое определение типов данных в расширенном наборе X
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = [c for c in X.columns if c not in numeric_features]

        # Разделение категорий на "особые" и "обычные"
        special_cat_features = [c for c in categorical_features if c in big_na_cols]
        regular_cat_features = [c for c in categorical_features if c not in big_na_cols]

        # Настройка трансформеров
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler", StandardScaler()),
        ])

        special_cat_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

        regular_cat_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

        # ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("special", special_cat_transformer, special_cat_features),
                ("regular", regular_cat_transformer, regular_cat_features),
            ],
            verbose_feature_names_out=False
        )
        return preprocessor