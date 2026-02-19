import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import VotingRegressor
#создание метрики huber loss
def huber_loss(y_true, y_pred, delta=0.1):
    error = y_true - y_pred
    is_small_error = np.abs(error) <= delta
    squared_loss = 0.5 * (error ** 2)
    linear_loss = delta * np.abs(error) - 0.5 * (delta ** 2)
    return np.where(is_small_error, squared_loss, linear_loss).mean()
#Универсальный класс чтобы удобно и быстро вытаскивать модели
class ModelFactory:
    _registry = {
        'linear_regression': LinearRegression,
        'lasso': Lasso,
        'ridge': Ridge,
        'elastic_net': ElasticNet,
        'xgboost': XGBRegressor,
        'lightgbm': LGBMRegressor,
        'catboost': CatBoostRegressor,
        'ensemble': VotingRegressor
    }

    @classmethod
    def get_model(cls, model_name, params=None):
        # Инициализация параметров пустым словарем, если они не переданы
        if params is None: params = {}
        if model_name not in cls._registry:
            raise ValueError(f"Модель {model_name} не найдена в реестре")
        # Возвращаем уже созданный объект модели
        return cls._registry[model_name](**params)

    @staticmethod # Статический метод для получения функции метрики и направления оптимизации
    def get_metric(metric_name):
        metric_name = metric_name.lower()
        if metric_name == 'rmse':
            return lambda y, p: np.sqrt(mean_squared_error(y, p)), 'minimize'
        elif metric_name == 'mse':
            return mean_squared_error, 'minimize'
        elif metric_name == 'mae':
            return mean_absolute_error, 'minimize'
        elif metric_name == 'r2':
            return r2_score, 'maximize'
        else:
            raise ValueError(f"Метрика {metric_name} не поддерживается")