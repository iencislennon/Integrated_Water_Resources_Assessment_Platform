"""
features.py
Формирование обучающей выборки: агрегация спутниковых признаков
к ячейкам GRACE (0.5° × 0.5°).
"""

import ee
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os

from data_loader import (
    get_grace_monthly, get_chirps_monthly, get_mod16_et, get_mod13_ndvi,
    get_train_region, get_basin_bbox, create_grace_grid, month_to_date
)

# ───────────────────────────────────────────────────────────────────────────────
# 1. Агрегация признаков к сетке GRACE
# ───────────────────────────────────────────────────────────────────────────────

def extract_features_for_month(
    year: int,
    month: int,
    grid: ee.FeatureCollection,
    region: ee.Geometry,
    scale_m: float = 500
) -> pd.DataFrame:
    """
    Для одного месяца: агрегировать CHIRPS, ET, NDVI к ячейкам сетки GRACE,
    добавить значение GRACE.

    Args:
        year, month: год и месяц
        grid: ee.FeatureCollection — сетка GRACE
        region: ee.Geometry — область
        scale_m: разрешение для reduceRegions (м)

    Returns:
        pd.DataFrame с колонками [year, month, lat, lon, precip_mean, et_mean, ndvi_mean, grace_lwe]
    """
    start_date = month_to_date(year, month)
    end_date = month_to_date(year, month + 1) if month < 12 else month_to_date(year + 1, 1)

    # --- GRACE ---
    grace = get_grace_monthly(start_date, end_date).first()

    # --- CHIRPS ---
    chirps = get_chirps_monthly(start_date, end_date).first()

    # --- ET ---
    et = get_mod16_et(start_date, end_date).first()

    # --- NDVI ---
    ndvi = get_mod13_ndvi(start_date, end_date).first()

    # --- Агрегация к сетке ---
    # Создаём композитный image со всеми признаками
    composite = ee.Image.cat([
        grace.select("lwe_thickness").rename("grace_lwe"),
        chirps.select("precipitation").rename("precip"),
        et.select("ET").rename("et"),
        ndvi.select("NDVI").rename("ndvi")
    ])

    # reduceRegions: среднее внутри каждой ячейки сетки
    stats = composite.reduceRegions(
        collection=grid,
        reducer=ee.Reducer.mean(),
        scale=scale_m,
        crs="EPSG:4326"
    )

    # Конвертируем в pandas
    # Используем getInfo (для небольших регионов) или экспорт в Asset
    features_list = stats.getInfo()["features"]

    rows = []
    for feat in features_list:
        props = feat["properties"]
        row = {
            "year": year,
            "month": month,
            "lat": props.get("lat"),
            "lon": props.get("lon"),
            "precip_mean": props.get("precip"),
            "et_mean": props.get("et"),
            "ndvi_mean": props.get("ndvi"),
            "grace_lwe": props.get("grace_lwe")
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


# ───────────────────────────────────────────────────────────────────────────────
# 2. Пакетная выгрузка через Earth Engine Export (для больших объёмов)
# ───────────────────────────────────────────────────────────────────────────────

def export_training_data_batch(
    years: list,
    months: list,
    grid: ee.FeatureCollection,
    region: ee.Geometry,
    output_dir: str = "data/processed",
    scale_m: float = 500,
    batch_size: int = 12
):
    """
    Пакетный экспорт обучающих данных через Earth Engine Export.table.toDrive.
    Более надёжно для больших объёмов, чем getInfo().

    Args:
        years: список годов
        months: список месяцев
        grid: сетка GRACE
        region: область
        output_dir: директория для сохранения
        scale_m: разрешение
        batch_size: сколько месяцев в одном экспорте

    Returns:
        Список task IDs
    """
    os.makedirs(output_dir, exist_ok=True)
    tasks = []

    for year in years:
        for month in months:
            start_date = month_to_date(year, month)
            end_date = month_to_date(year, month + 1) if month < 12 else month_to_date(year + 1, 1)

            # Собираем композит
            grace = get_grace_monthly(start_date, end_date).first()
            chirps = get_chirps_monthly(start_date, end_date).first()
            et = get_mod16_et(start_date, end_date).first()
            ndvi = get_mod13_ndvi(start_date, end_date).first()

            composite = ee.Image.cat([
                grace.select("lwe_thickness").rename("grace_lwe"),
                chirps.select("precipitation").rename("precip"),
                et.select("ET").rename("et"),
                ndvi.select("NDVI").rename("ndvi")
            ])

            # reduceRegions
            stats = composite.reduceRegions(
                collection=grid,
                reducer=ee.Reducer.mean(),
                scale=scale_m,
                crs="EPSG:4326"
            )

            # Добавляем year и month как свойства
            stats = stats.map(lambda f: f.set("year", year, "month", month))

            # Экспорт в CSV на Google Drive
            task = ee.batch.Export.table.toDrive(
                collection=stats,
                description=f"training_{year}_{month:02d}",
                folder="ili_balkhash_training",
                fileFormat="CSV"
            )
            task.start()
            tasks.append(task.id)

            print(f"🚀 Запущен экспорт: {year}-{month:02d} (task: {task.id})")

    return tasks


# ───────────────────────────────────────────────────────────────────────────────
# 3. Сборка CSV из локальных файлов
# ───────────────────────────────────────────────────────────────────────────────

def combine_csv_files(csv_dir: str, output_path: str = "data/processed/training_data.csv"):
    """
    Объединить все CSV-файлы из директории в один training_data.csv.

    Args:
        csv_dir: директория с CSV-файлами (скачанными из Google Drive)
        output_path: путь к итоговому CSV
    """
    import glob

    csv_files = sorted(glob.glob(os.path.join(csv_dir, "training_*.csv")))

    if not csv_files:
        raise FileNotFoundError(f"CSV-файлы не найдены в {csv_dir}")

    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    # Переименовываем колонки для единообразия
    column_map = {
        "precip": "precip_mean",
        "et": "et_mean",
        "ndvi": "ndvi_mean",
        "grace_lwe": "grace_lwe",
        "latitude": "lat",
        "longitude": "lon"
    }
    combined = combined.rename(columns={k: v for k, v in column_map.items() if k in combined.columns})

    # Сохраняем
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    combined.to_csv(output_path, index=False)

    print(f"✅ Объединено {len(csv_files)} файлов → {output_path}")
    print(f"   Записей: {len(combined)}, период: {combined['year'].min()}-{combined['month'].min()} — {combined['year'].max()}-{combined['month'].max()}")

    return combined


# ───────────────────────────────────────────────────────────────────────────────
# 4. Предобработка (очистка, фильтрация выбросов)
# ───────────────────────────────────────────────────────────────────────────────

def preprocess_training_data(
    df: pd.DataFrame,
    iqr_multiplier: float = 3.0
) -> pd.DataFrame:
    """
    Очистка обучающих данных:
    1. Удаление строк с NaN
    2. Фильтрация выбросов по IQR для grace_lwe
    3. Добавление временных признаков (sin/cos месяца)

    Args:
        df: DataFrame с сырыми данными
        iqr_multiplier: множитель для IQR (3.0 = очень консервативно)

    Returns:
        Очищенный DataFrame
    """
    print(f"Исходных записей: {len(df)}")

    # Удаляем NaN
    df_clean = df.dropna().copy()
    print(f"После удаления NaN: {len(df_clean)}")

    # Фильтрация выбросов по IQR для grace_lwe
    q1 = df_clean["grace_lwe"].quantile(0.25)
    q3 = df_clean["grace_lwe"].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - iqr_multiplier * iqr
    upper = q3 + iqr_multiplier * iqr

    df_clean = df_clean[
        (df_clean["grace_lwe"] >= lower) & 
        (df_clean["grace_lwe"] <= upper)
    ].copy()
    print(f"После фильтрации выбросов: {len(df_clean)}")

    # Временные признаки (циклические)
    df_clean["month_sin"] = np.sin(2 * np.pi * df_clean["month"] / 12)
    df_clean["month_cos"] = np.cos(2 * np.pi * df_clean["month"] / 12)

    # Дата для удобства
    df_clean["date"] = pd.to_datetime(df_clean[["year", "month"]].assign(day=1))

    return df_clean


# ───────────────────────────────────────────────────────────────────────────────
# 5. Генерация признаков для предсказания (500 м)
# ───────────────────────────────────────────────────────────────────────────────

def create_prediction_features(
    year: int,
    month: int,
    region: ee.Geometry,
    scale_m: float = 500
) -> ee.Image:
    """
    Создать композитный image с признаками для предсказания на разрешении 500 м.

    Args:
        year, month: год и месяц
        region: область предсказания (бассейн Или-Балхаш)
        scale_m: целевое разрешение

    Returns:
        ee.Image с bands: [precip, et, ndvi, month_sin, month_cos]
    """
    start_date = month_to_date(year, month)
    end_date = month_to_date(year, month + 1) if month < 12 else month_to_date(year + 1, 1)

    # Загружаем данные
    chirps = get_chirps_monthly(start_date, end_date).first()
    et = get_mod16_et(start_date, end_date).first()
    ndvi = get_mod13_ndvi(start_date, end_date).first()

    # Ресемплим все к одному разрешению (500 м)
    # Используем CHIRPS (~5.5 км) как базу для ресемплинга осадков
    # ET и NDVI уже ~500 м, но ресемплим для единообразия

    base = ee.Image.constant(0).clip(region)

    precip_resampled = chirps.select("precipitation").resample("bilinear").reproject(
        crs="EPSG:4326", scale=scale_m
    ).rename("precip")

    et_resampled = et.select("ET").resample("bilinear").reproject(
        crs="EPSG:4326", scale=scale_m
    ).rename("et")

    ndvi_resampled = ndvi.select("NDVI").resample("bilinear").reproject(
        crs="EPSG:4326", scale=scale_m
    ).rename("ndvi")

    # Временные признаки как константные bands
    month_sin = ee.Image.constant(np.sin(2 * np.pi * month / 12)).rename("month_sin")
    month_cos = ee.Image.constant(np.cos(2 * np.pi * month / 12)).rename("month_cos")

    # Комбинируем
    features = ee.Image.cat([
        precip_resampled,
        et_resampled,
        ndvi_resampled,
        month_sin,
        month_cos
    ]).clip(region)

    return features


if __name__ == "__main__":
    print("Тест features.py")
    print("Все функции загружены!")
