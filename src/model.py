"""
model.py
Обучение RandomForest для даунскейлинга GRACE,
применение модели к растру 500 м.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Признаки для реальных (не демо) данных: precip/et/ndvi + месяц циклически,
# координаты ячейки (lat, lon) и лаги 1-3 мес. (см. features.py). lat/lon
# позволяют модели различать регионы внутри широкой обучающей области
# (Казахстан + Центральная Азия), лаги precip/et/ndvi — учитывают задержку
# гидрологического отклика (регулируемый сброс Капшагайского водохранилища).
#
# Это "честный" даунскейлинг: предсказание строится только из спутниковых
# прокси-признаков, без доступа к самой GRACE за прошлый месяц.
DOWNSCALING_FEATURE_COLS_NO_AR = [
    "precip_mean", "et_mean", "ndvi_mean", "month_sin", "month_cos",
    "lat", "lon",
    "precip_mean_lag1", "precip_mean_lag2", "precip_mean_lag3",
    "et_mean_lag1", "et_mean_lag2", "et_mean_lag3",
    "ndvi_mean_lag1", "ndvi_mean_lag2", "ndvi_mean_lag3",
]

# То же самое + авторегрессионный лаг GRACE (grace_lwe за месяц T-1). Резко
# поднимает test R^2 (0.40 -> 0.95) и корреляцию с DAHITI (0.40 -> 0.67), НО:
# GRACE не имеет разрешения тоньше mascon (~55 км) — grace_lwe_lag1 одинаков
# для всех ячеек 0.5° внутри одного mascon, так что прирост в основном
# отражает персистентность сигнала (t-1 ≈ t), а не улучшение пространственного
# даунскейлинга внутри бассейна. Использовать с этой оговоркой в отчётах.
DOWNSCALING_FEATURE_COLS = DOWNSCALING_FEATURE_COLS_NO_AR + ["grace_lwe_lag1"]

# ───────────────────────────────────────────────────────────────────────────────
# 1. Обучение модели
# ───────────────────────────────────────────────────────────────────────────────

def train_model(
    df: pd.DataFrame,
    feature_cols: list = None,
    target_col: str = "grace_lwe",
    train_years: tuple = (2002, 2020),
    test_years: tuple = (2021, 2024),
    model_params: dict = None,
    model_path: str = "models/rf_grace_downscaling.pkl"
):
    """
    Обучить RandomForest на агрегированных данных.

    Args:
        df: DataFrame с обучающими данными
        feature_cols: список признаков (по умолчанию все кроме target)
        target_col: целевая переменная
        train_years: (start, end) для train
        test_years: (start, end) для test
        model_params: параметры RandomForest
        model_path: куда сохранить модель

    Returns:
        dict с model, metrics, feature_importance
    """
    # Признаки по умолчанию
    if feature_cols is None:
        feature_cols = DOWNSCALING_FEATURE_COLS

    # Параметры модели по умолчанию
    if model_params is None:
        model_params = {
            "n_estimators": 200,
            "max_depth": 15,
            "random_state": 42,
            "n_jobs": -1,
            "min_samples_split": 5,
            "min_samples_leaf": 2
        }

    # Разделение на train/test по годам (временное!)
    train_mask = (df["year"] >= train_years[0]) & (df["year"] <= train_years[1])
    test_mask = (df["year"] >= test_years[0]) & (df["year"] <= test_years[1])

    df_train = df[train_mask].copy()
    df_test = df[test_mask].copy()

    X_train = df_train[feature_cols].values
    y_train = df_train[target_col].values
    X_test = df_test[feature_cols].values
    y_test = df_test[target_col].values

    print(f"Train: {len(df_train)} записей ({train_years[0]}-{train_years[1]})")
    print(f"Test:  {len(df_test)} записей ({test_years[0]}-{test_years[1]})")

    # Обучение
    model = RandomForestRegressor(**model_params)
    model.fit(X_train, y_train)

    # Предсказания
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Метрики
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print(f"\n📊 Метрики:")
    print(f"   Train RMSE: {train_rmse:.4f} | R²: {train_r2:.4f}")
    print(f"   Test  RMSE: {test_rmse:.4f} | R²: {test_r2:.4f}")

    # Важность признаков
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    print(f"\n🔍 Важность признаков:")
    for _, row in importance.iterrows():
        print(f"   {row['feature']}: {row['importance']:.4f}")

    # Сохранение модели
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"\n💾 Модель сохранена: {model_path}")

    return {
        "model": model,
        "metrics": {
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
            "train_r2": train_r2,
            "test_r2": test_r2
        },
        "feature_importance": importance,
        "model_path": model_path
    }


# ───────────────────────────────────────────────────────────────────────────────
# 2. Загрузка модели
# ───────────────────────────────────────────────────────────────────────────────

def load_model(model_path: str = "models/rf_grace_downscaling.pkl"):
    """Загрузить сохранённую модель."""
    model = joblib.load(model_path)
    print(f"✅ Модель загружена: {model_path}")
    return model


# ───────────────────────────────────────────────────────────────────────────────
# 3. Предсказание на растре (Earth Engine + numpy)
# ───────────────────────────────────────────────────────────────────────────────

def predict_raster_ee(
    model,
    features_image: "ee.Image",
    region: "ee.Geometry",
    scale_m: float = 500,
    output_asset_id: str = None
):
    """
    Применить модель RandomForest к ee.Image с признаками.

    Используем подход: экспортируем признаки как numpy array,
    предсказываем, загружаем обратно как ee.Image.

    Args:
        model: обученный sklearn модель
        features_image: ee.Image с bands [precip, et, ndvi, month_sin, month_cos]
        region: область
        scale_m: разрешение
        output_asset_id: ID для сохранения в Earth Engine Assets (опционально)

    Returns:
        ee.Image с предсказанными значениями
    """
    import ee

    # Экспортируем признаки как numpy array через sample
    # Альтернатива: Export.image.toDrive → GeoTIFF → numpy

    # Для демо-прототипа используем Earth Engine's array-based approach
    # или экспортируем в numpy

    # Вариант 1: Используем Earth Engine's array image для предсказания
    # (ограничен, т.к. sklearn не работает в EE напрямую)

    # Вариант 2: Экспорт в numpy, предсказание, импорт обратно

    # Для простоты — используем geemap для экспорта
    print("Экспорт признаков в numpy array...")

    # Получаем массив значений
    # Сэмплируем регулярную сетку
    sample = features_image.sampleRectangle(region=region, defaultValue=0)

    # Получаем numpy arrays для каждого band
    bands = ["precip", "et", "ndvi", "month_sin", "month_cos"]
    arrays = {}
    for band in bands:
        arr = np.array(sample.get(band).getInfo())
        arrays[band] = arr

    # Формируем X для предсказания
    # [n_pixels, n_features]
    shape = arrays["precip"].shape
    n_pixels = shape[0] * shape[1]

    X = np.column_stack([
        arrays["precip"].flatten(),
        arrays["et"].flatten(),
        arrays["ndvi"].flatten(),
        arrays["month_sin"].flatten(),
        arrays["month_cos"].flatten()
    ])

    # Предсказываем
    print("Предсказание...")
    y_pred = model.predict(X)

    # Возвращаем в форму растра
    pred_raster = y_pred.reshape(shape)

    # Создаём ee.Image из numpy
    # Получаем проекцию из исходного image
    projection = features_image.projection()

    # Создаём image из numpy array
    pred_image = ee.Image.arrayConvert(
        ee.Array(pred_raster.tolist()),
        0
    ).reproject(projection).rename("predicted_grace")

    # Клиппинг по региону
    pred_image = pred_image.clip(region)

    # Сохранение в Asset (опционально)
    if output_asset_id:
        task = ee.batch.Export.image.toAsset(
            image=pred_image,
            description="downscaled_grace",
            assetId=output_asset_id,
            region=region,
            scale=scale_m,
            crs="EPSG:4326"
        )
        task.start()
        print(f"🚀 Экспорт в Asset запущен: {output_asset_id}")

    return pred_image


# ───────────────────────────────────────────────────────────────────────────────
# 4. Предсказание через GeoTIFF (альтернатива)
# ───────────────────────────────────────────────────────────────────────────────

def predict_raster_geotiff(
    model,
    geotiff_path: str,
    output_path: str,
    feature_bands: list = None
):
    """
    Предсказание на GeoTIFF файле (локально через rasterio).

    Args:
        model: обученный sklearn модель
        geotiff_path: путь к GeoTIFF с признаками (multi-band)
        output_path: путь для сохранения предсказаний
        feature_bands: список названий band-ов (по умолчанию 1-5)
    """
    try:
        import rasterio
    except ImportError:
        raise ImportError("Установите rasterio: pip install rasterio")

    if feature_bands is None:
        feature_bands = [1, 2, 3, 4, 5]

    with rasterio.open(geotiff_path) as src:
        # Читаем все bands
        bands_data = []
        for b in feature_bands:
            bands_data.append(src.read(b))

        profile = src.profile
        shape = bands_data[0].shape

        # Формируем X
        X = np.column_stack([b.flatten() for b in bands_data])

        # Предсказываем
        y_pred = model.predict(X)
        pred_raster = y_pred.reshape(shape)

        # Сохраняем
        profile.update(
            count=1,
            dtype=rasterio.float32,
            compress="lzw"
        )

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(pred_raster.astype(rasterio.float32), 1)

    print(f"✅ Предсказания сохранены: {output_path}")
    return output_path


# ───────────────────────────────────────────────────────────────────────────────
# 5. Пакетное предсказание для диапазона дат
# ───────────────────────────────────────────────────────────────────────────────

def predict_batch(
    model,
    years: list,
    months: list,
    region: "ee.Geometry",
    output_dir: str = "data/processed/predictions_500m",
    scale_m: float = 500,
    export_to_drive: bool = True
):
    """
    Пакетное предсказание для набора месяцев.

    Args:
        model: обученная модель
        years: список годов
        months: список месяцев
        region: область
        output_dir: директория для сохранения
        scale_m: разрешение
        export_to_drive: экспортировать в Google Drive

    Returns:
        Список task IDs
    """
    from features import create_prediction_features
    import ee

    os.makedirs(output_dir, exist_ok=True)
    tasks = []

    for year in years:
        for month in months:
            print(f"\n🔄 Обработка: {year}-{month:02d}")

            # Создаём признаки
            features = create_prediction_features(year, month, region, scale_m)

            # Экспортируем признаки как GeoTIFF
            feature_asset = f"projects/PROJECT_ID/assets/features_{year}_{month:02d}"

            # Экспорт признаков
            task_feat = ee.batch.Export.image.toDrive(
                image=features,
                description=f"features_{year}_{month:02d}",
                folder="ili_balkhash_features",
                region=region,
                scale=scale_m,
                crs="EPSG:4326",
                fileFormat="GeoTIFF"
            )
            task_feat.start()

            # Предсказание будет выполнено локально после скачивания
            # (см. predict_raster_geotiff)

            tasks.append({
                "year": year,
                "month": month,
                "feature_task": task_feat.id
            })

            print(f"   🚀 Экспорт признаков запущен")

    return tasks


if __name__ == "__main__":
    print("Тест model.py")
    print("Все функции загружены!")
