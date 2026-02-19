import os
import sys
import io
import warnings
import pathlib
import yaml
import numpy as np
import pandas as pd
import optuna
import mlflow
import mlflow.sklearn
from sklearn.model_selection import KFold, cross_val_score, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import make_scorer
warnings.filterwarnings('ignore')
# Настройка UTF-8 для Windows (у меня были странные символы в терминале:) )
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
from data_loader import DataLoader
from model_factory import ModelFactory, huber_loss
from logger import Logger
from preprocessor import HouseFeatureEngineer, PreprocessorBuilder
# Определение путей
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)
log = Logger()
# Загрузка конфигурации из YAML
def load_config(path="config/config.yaml"):
    config_path = os.path.join(project_root, path)
    if not os.path.exists(config_path):
        log.error(f"Файл конфигурации не найден: {config_path}")
        raise FileNotFoundError(config_path)
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

# Парсинг параметров для Optuna
def parse_optuna_params(trial, model_name, config_models):
    raw_params = config_models.get(model_name, {})
    params = {}
    for param_name, values in raw_params.items():
        if isinstance(values, list):
            p_type = values[0]
            if p_type == 'int':
                params[param_name] = trial.suggest_int(param_name, values[1], values[2])
            elif p_type == 'float':
                params[param_name] = trial.suggest_float(param_name, values[1], values[2], 
                                                         log=values[3].get('log', False) if len(values)>3 else False)
        else:
            params[param_name] = values
    return params

# Целевая функция для Optuna
def objective(trial, X, y, cfg):
    model_name = cfg["run"]["model_name"]
    metric_name = cfg["run"]["metrics"]
    
    # 1. Создание модели или ансамбля
    if model_name == "ensemble":
        ensemble_models = cfg["ensemble"]["models"]
        weights = [trial.suggest_float(f"weight_{m}", 0.0, 1.0) for m in ensemble_models]
        estimators = []
        for m_name in ensemble_models:
            m_params = cfg["models"].get(m_name, {})
            clean_params = {k: (v[1] if isinstance(v, list) else v) for k, v in m_params.items()}
            estimators.append((m_name, ModelFactory.get_model(m_name, clean_params)))
        model = VotingRegressor(estimators=estimators, weights=weights)
    else:
        params = parse_optuna_params(trial, model_name, cfg["models"])
        model = ModelFactory.get_model(model_name, params)
    
    # 2. Сборка препроцессора 
    preprocessor = PreprocessorBuilder.build(X)
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # 3. Кросс-валидация
    cv = KFold(n_splits=5, shuffle=True, random_state=cfg['base']['seed'])
    scoring_map = {'rmse': 'neg_root_mean_squared_error', 'r2': 'r2'}
    metric = scoring_map.get(metric_name.lower(), 'neg_root_mean_squared_error')
    
    try:
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring=metric, n_jobs=1)
        mean_score = scores.mean()
        return -mean_score if 'neg_' in metric else mean_score
    except Exception as e:
        log.error(f"Ошибка в Trial: {e}")
        return float('inf')

def main():
    log.section("Запуск Pipeline House Pricing") #
    
    cfg = load_config()
    model_name = cfg["run"]["model_name"]
    target_col = cfg["base"]["target"]
    
    log.info(f"Архитектура: {model_name} | Оптимизация по: {cfg['run']['metrics']}")

    # 1. Загрузка данных через DataLoader
    train_df = DataLoader.load_csv(os.path.join(project_root, cfg["base"]["train_data"]))
    test_df = DataLoader.load_csv(os.path.join(project_root, "data/test.csv"))

    # Очистка и логарифмирование целевой переменной
    train_df = train_df.dropna(subset=[target_col])
    y = np.log1p(train_df[target_col])
    
    X_raw = train_df.drop(columns=[target_col, 'Id'], errors='ignore')
    X_test_raw = test_df.drop(columns=["Id"], errors='ignore')
    X_test_ids = test_df["Id"]

    # 2. ПРИМЕНЕНИЕ FEATURE ENGINEERING 
    log.info("Генерация признаков...")
    fe = HouseFeatureEngineer()
    X = fe.transform(X_raw) 
    X_test = fe.transform(X_test_raw) 

    # 3. Поиск гиперпараметров через Optuna
    log.section("Оптимизация (Optuna)")
    _, direction = ModelFactory.get_metric(cfg["run"]["metrics"])
    study = optuna.create_study(direction=direction, study_name=f"{model_name}_adv_optimization")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    study.optimize(lambda trial: objective(trial, X, y, cfg), 
                   n_trials=cfg["run"]["n_trials"], show_progress_bar=True)
    
    log.info("Оптимизация завершена.")

    # 4. Финальная оценка и MLflow
    mlflow_dir = os.path.join(project_root, "mlruns")
    mlflow.set_tracking_uri(pathlib.Path(mlflow_dir).as_uri())
    mlflow.set_experiment(cfg["base"].get("experiment_name", "House_Pricing_Advanced"))
    
    log.section("Финальная оценка и запись в MLflow")
    
    with mlflow.start_run(run_name=f"best_{model_name}_with_FE"):
        
        # Сборка финальной модели
        if model_name == "ensemble":
            estimators = []
            best_weights = [study.best_params[f"weight_{m}"] for m in cfg["ensemble"]["models"]]
            for m in cfg["ensemble"]["models"]:
                m_params = cfg["models"].get(m, {})
                clean_params = {k: (v[1] if isinstance(v, list) else v) for k, v in m_params.items()}
                estimators.append((m, ModelFactory.get_model(m, clean_params)))
            final_model = VotingRegressor(estimators=estimators, weights=best_weights)
            mlflow.log_params(study.best_params)
        else:
            final_model = ModelFactory.get_model(model_name, study.best_params)
            mlflow.log_params(study.best_params)

        preprocessor = PreprocessorBuilder.build(X)
        
        final_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', final_model)
        ])

        # Расчет итоговых метрик на кросс-валидации
        huber_scorer = make_scorer(huber_loss, greater_is_better=False)
        scoring = {'MAE': 'neg_mean_absolute_error', 'RMSE': 'neg_root_mean_squared_error', 'R2': 'r2'}
        cv_results = cross_validate(final_pipeline, X, y, cv=5, scoring=scoring)
        
        metrics = {
            'RMSE': -cv_results['test_RMSE'].mean(),
            'MAE': -cv_results['test_MAE'].mean(),
            'R2': cv_results['test_R2'].mean()
        }

        for name, val in metrics.items():
            log.metric(name, val)
        mlflow.log_metrics(metrics)

        # 5. Обучение Production-модели и сохранение
        log.section("Сохранение результатов")
        final_pipeline.fit(X, y)
        mlflow.sklearn.log_model(final_pipeline, name="model")
        
        # Предсказание на тесте 
        preds = np.expm1(final_pipeline.predict(X_test))
        submission = pd.DataFrame({"Id": X_test_ids, "SalePrice": preds})
        
        out_path = os.path.join(project_root, cfg["base"]["log_dir"], f"submission_{model_name}_fe.csv")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        submission.to_csv(out_path, index=False)
        
        mlflow.log_artifact(out_path)
        log.info(f"Файл готов к отправке на Kaggle: {out_path}")

if __name__ == "__main__":
    main()