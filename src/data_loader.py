"""
data_loader.py
Загрузка и агрегация данных из Google Earth Engine и DAHITI API.
"""

import ee
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os

# ───────────────────────────────────────────────────────────────────────────────
# Инициализация Earth Engine
# ───────────────────────────────────────────────────────────────────────────────

def init_ee(project_id: str = None):
    """Инициализация Earth Engine."""
    try:
        ee.Initialize(project=project_id)
        print("✅ Earth Engine инициализирован")
    except Exception as e:
        print(f"⚠️ Ошибка инициализации: {e}")
        print("Запускаем аутентификацию...")
        ee.Authenticate()
        ee.Initialize(project=project_id)

# ───────────────────────────────────────────────────────────────────────────────
# Регионы интереса
# ───────────────────────────────────────────────────────────────────────────────

# Область обучения: Казахстан + Центральная Азия
TRAIN_BBOX = {
    'west': 45.0, 'south': 40.0,
    'east': 90.0, 'north': 55.0
}

# Суб-бассейн Или-Балхаш (приблизительный bbox)
BASIN_BBOX = {
    'west': 74.0, 'south': 43.0,
    'east': 82.0, 'north': 48.0
}

# Центр оз. Балхаш (для маркера на карте)
BALKHASH_CENTER = [46.5, 78.0]

# ───────────────────────────────────────────────────────────────────────────────
# Earth Engine коллекции
# ───────────────────────────────────────────────────────────────────────────────

GRACE_COLLECTION = "NASA/GRACE/MASS_GRIDS_V04/MASCON"
CHIRPS_COLLECTION = "UCSB-CHC/CHIRPS/V3/PENTAD"
MOD16_ET_COLLECTION = "MODIS/061/MOD16A2GF"  # gap-filled
MOD16_ET_NRT_COLLECTION = "MODIS/061/MOD16A2"  # near-real-time
MOD13_NDVI_COLLECTION = "MODIS/061/MOD13Q1"
HYDROSHEDS_BASIN = "WWF/HydroSHEDS/v1/Basins/hybas_6"

# ───────────────────────────────────────────────────────────────────────────────
# 1. Граница бассейна
# ───────────────────────────────────────────────────────────────────────────────

def get_basin_polygon(basin_level: int = 6, basin_id: int = None):
    """
    Получить полигон бассейна Или-Балхаш из HydroSHEDS.

    Args:
        basin_level: уровень HydroSHEDS (5, 6, 7)
        basin_id: конкретный ID бассейна (если известен)

    Returns:
        ee.Feature — полигон бассейна
    """
    collection = f"WWF/HydroSHEDS/v1/Basins/hybas_{basin_level}"
    basins = ee.FeatureCollection(collection)

    # Фильтруем по bbox бассейна Или-Балхаш
    basin_fc = basins.filterBounds(ee.Geometry.Rectangle([
        BASIN_BBOX['west'], BASIN_BBOX['south'],
        BASIN_BBOX['east'], BASIN_BBOX['north']
    ]))

    # Если задан конкретный ID — фильтруем по нему
    if basin_id is not None:
        basin_fc = basin_fc.filter(ee.Filter.eq("HYBAS_ID", basin_id))

    # Берём первый (или объединяем все, если несколько)
    basin = basin_fc.first()

    if basin is None:
        raise ValueError(f"Бассейн не найден в HydroSHEDS hybas_{basin_level}")

    return basin


def get_basin_bbox():
    """Вернуть ee.Geometry.Rectangle для бассейна Или-Балхаш."""
    return ee.Geometry.Rectangle([
        BASIN_BBOX['west'], BASIN_BBOX['south'],
        BASIN_BBOX['east'], BASIN_BBOX['north']
    ])


def get_train_region():
    """Вернуть ee.Geometry.Rectangle для региона обучения."""
    return ee.Geometry.Rectangle([
        TRAIN_BBOX['west'], TRAIN_BBOX['south'],
        TRAIN_BBOX['east'], TRAIN_BBOX['north']
    ])

# ───────────────────────────────────────────────────────────────────────────────
# 2. GRACE данные
# ───────────────────────────────────────────────────────────────────────────────

def get_grace_collection(start_date: str = "2002-04-01", end_date: str = None):
    """
    Получить коллекцию GRACE/GRACE-FO Mascon.

    Args:
        start_date: начальная дата (YYYY-MM-DD)
        end_date: конечная дата (YYYY-MM-DD), по умолчанию — сегодня

    Returns:
        ee.ImageCollection
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    grace = ee.ImageCollection(GRACE_COLLECTION)\
        .select("lwe_thickness")\
        .filterDate(start_date, end_date)

    return grace


def get_grace_monthly(start_date: str = "2002-04-01", end_date: str = None):
    """
    Агрегировать GRACE в месячные средние.
    GRACE Mascon уже имеет месячное разрешение, но на всякий случай — mean.
    """
    grace = get_grace_collection(start_date, end_date)

    # GRACE Mascon — уже месячные, просто возвращаем
    return grace

# ───────────────────────────────────────────────────────────────────────────────
# 3. CHIRPS осадки → месячные суммы
# ───────────────────────────────────────────────────────────────────────────────

def get_chirps_monthly(start_date: str = "2002-04-01", end_date: str = None):
    """
    Агрегировать CHIRPS пентады в месячные суммы осадков.

    Returns:
        ee.ImageCollection — месячные суммы осадков (мм/месяц)
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    chirps = ee.ImageCollection(CHIRPS_COLLECTION)\
        .select("precipitation")\
        .filterDate(start_date, end_date)

    # Агрегируем по месяцам: сумма пентад = сумма осадков за месяц
    def monthly_sum(year, month):
        start = ee.Date.fromYMD(year, month, 1)
        end = start.advance(1, "month")
        monthly = chirps.filterDate(start, end).sum()
        return monthly.set({
            "year": year,
            "month": month,
            "system:time_start": start.millis()
        })

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    years_list = list(range(start_dt.year, end_dt.year + 1))

    years = ee.List(years_list)
    months = ee.List.sequence(1, 12)

    monthly_ic = ee.ImageCollection.fromImages(
        years.map(lambda y: months.map(lambda m: monthly_sum(y, m))).flatten()
    )

    return monthly_ic.filterDate(start_date, end_date)

# ───────────────────────────────────────────────────────────────────────────────
# 4. MODIS ET
# ───────────────────────────────────────────────────────────────────────────────

def get_mod16_et(start_date: str = "2002-04-01", end_date: str = None, 
                 use_gf: bool = True):
    """
    Получить эвапотранспирацию MOD16.

    Args:
        use_gf: True — gap-filled (MOD16A2GF, с задержкой ~1 год),
                False — near-real-time (MOD16A2, с пропусками)

    Returns:
        ee.ImageCollection — ET (кг/м²/8дн = мм/8дн)
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    collection = MOD16_ET_COLLECTION if use_gf else MOD16_ET_NRT_COLLECTION

    et = ee.ImageCollection(collection)\
        .select("ET")\
        .filterDate(start_date, end_date)

    # Агрегируем в месячные средние
    def monthly_mean(year, month):
        start = ee.Date.fromYMD(year, month, 1)
        end = start.advance(1, "month")
        monthly = et.filterDate(start, end).mean()
        return monthly.set({
            "year": year,
            "month": month,
            "system:time_start": start.millis()
        })

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    years_list = list(range(start_dt.year, end_dt.year + 1))

    years = ee.List(years_list)
    months = ee.List.sequence(1, 12)

    monthly_ic = ee.ImageCollection.fromImages(
        years.map(lambda y: months.map(lambda m: monthly_mean(y, m))).flatten()
    )

    return monthly_ic.filterDate(start_date, end_date)

# ───────────────────────────────────────────────────────────────────────────────
# 5. MODIS NDVI
# ───────────────────────────────────────────────────────────────────────────────

def get_mod13_ndvi(start_date: str = "2002-04-01", end_date: str = None):
    """
    Получить NDVI MOD13Q1, агрегированный в месячные средние.

    Returns:
        ee.ImageCollection — месячный средний NDVI
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    ndvi = ee.ImageCollection(MOD13_NDVI_COLLECTION)\
        .select("NDVI")\
        .filterDate(start_date, end_date)

    # NDVI в MOD13Q1 масштабирован ×10000, нормализуем
    def normalize(image):
        return image.divide(10000).copyProperties(image, ["system:time_start"])

    ndvi = ndvi.map(normalize)

    # Агрегируем в месячные средние
    def monthly_mean(year, month):
        start = ee.Date.fromYMD(year, month, 1)
        end = start.advance(1, "month")
        monthly = ndvi.filterDate(start, end).mean()
        return monthly.set({
            "year": year,
            "month": month,
            "system:time_start": start.millis()
        })

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    years_list = list(range(start_dt.year, end_dt.year + 1))

    years = ee.List(years_list)
    months = ee.List.sequence(1, 12)

    monthly_ic = ee.ImageCollection.fromImages(
        years.map(lambda y: months.map(lambda m: monthly_mean(y, m))).flatten()
    )

    return monthly_ic.filterDate(start_date, end_date)

# ───────────────────────────────────────────────────────────────────────────────
# 6. DAHITI API — уровень оз. Балхаш
# ───────────────────────────────────────────────────────────────────────────────

DAHITI_BASE_URL = "https://dahiti.dgfi.tum.de/api/v2/"


def fetch_dahiti_water_level(api_key: str,
                              dahiti_id: int = 91,
                              output_path: str = "data/raw/dahiti_balkhash.json"):
    """
    Загрузить данные об уровне воды оз. Балхаш из DAHITI API v2.

    DAHITI API v2 аутентифицируется персональным api_key, а не логином/паролем
    от сайта (см. https://dahiti.dgfi.tum.de/en/frequently-asked-questions/api-key/).

    Args:
        api_key: персональный DAHITI API-ключ
        dahiti_id: ID оз. Балхаш в DAHITI (91, см. https://dahiti.dgfi.tum.de/en/91/)
        output_path: куда сохранить JSON

    Returns:
        pd.DataFrame с колонками [date, water_level]
    """
    params = {
        "api_key": api_key,
        "dahiti_id": dahiti_id,
        "format": "json"
    }

    response = requests.get(DAHITI_BASE_URL + "download-water-level/", params=params, timeout=60)
    response.raise_for_status()

    data = response.json()

    if isinstance(data, dict) and data.get("statuscode") not in (None, 200):
        raise ValueError(f"DAHITI API вернул ошибку: {data.get('message', data)}")

    # Преобразуем в DataFrame
    if isinstance(data, list):
        df = pd.DataFrame(data)
    elif isinstance(data, dict) and "data" in data:
        df = pd.DataFrame(data["data"])
    else:
        df = pd.DataFrame(data)

    # DAHITI v2 возвращает "wse" (water surface elevation) вместо "water_level"
    df = df.rename(columns={"wse": "water_level", "wse_u": "water_level_uncertainty"})

    # Приводим даты
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    elif "datetime" in df.columns:
        df["date"] = pd.to_datetime(df["datetime"])

    # Сортируем по дате
    df = df.sort_values("date").reset_index(drop=True)

    # Сохраняем в едином плоском формате [date, water_level, ...],
    # совместимом с load_dahiti_local() и dashboard
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_json(output_path, orient="records", date_format="iso")

    print(f"✅ DAHITI: загружено {len(df)} записей, период {df['date'].min()} — {df['date'].max()}")

    return df


def load_dahiti_local(path: str = "data/raw/dahiti_balkhash.json"):
    """Загрузить сохранённые данные DAHITI из локального JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        df = pd.DataFrame(data)
    elif isinstance(data, dict) and "data" in data:
        df = pd.DataFrame(data["data"])
    else:
        df = pd.DataFrame(data)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    elif "datetime" in df.columns:
        df["date"] = pd.to_datetime(df["datetime"])

    df = df.sort_values("date").reset_index(drop=True)
    return df


# ───────────────────────────────────────────────────────────────────────────────
# 7. Создание сетки GRACE (0.5° × 0.5°)
# ───────────────────────────────────────────────────────────────────────────────

def create_grace_grid(region: ee.Geometry, scale_deg: float = 0.5):
    """
    Создать регулярную сетку с заданным шагом в градусах.

    Args:
        region: ee.Geometry — область для сетки
        scale_deg: шаг сетки в градусах (GRACE = 0.5°)

    Returns:
        ee.FeatureCollection — каждая ячейка сетки как Feature с lat/lon
    """
    # Создаём пиксельную сетку
    lon_lat = ee.Image.pixelLonLat()

    # Округляем координаты до шага сетки
    lon = lon_lat.select("longitude").multiply(1 / scale_deg).floor().multiply(scale_deg)
    lat = lon_lat.select("latitude").multiply(1 / scale_deg).floor().multiply(scale_deg)

    grid_image = lon.addBands(lat).rename(["lon", "lat"])

    # Сэмплируем в точки
    grid_points = grid_image.sample(
        region=region,
        scale=scale_deg * 111320,  # примерно метры на градус
        geometries=True
    )

    # Преобразуем точки в ячейки (прямоугольники)
    def point_to_cell(feature):
        lon = feature.get("lon")
        lat = feature.get("lat")

        cell = ee.Geometry.Rectangle([
            lon, lat,
            ee.Number(lon).add(scale_deg),
            ee.Number(lat).add(scale_deg)
        ])

        return ee.Feature(cell).set({
            "lon": lon,
            "lat": lat,
            "cell_id": ee.String(lon).cat("_").cat(ee.String(lat))
        })

    grid = grid_points.map(point_to_cell)

    # Убираем дубликаты по cell_id
    grid = grid.distinct("cell_id")

    return grid


# ───────────────────────────────────────────────────────────────────────────────
# 8. Утилиты
# ───────────────────────────────────────────────────────────────────────────────

def month_to_date(year: int, month: int) -> str:
    """Преобразовать год и месяц в строку YYYY-MM-DD."""
    return f"{year:04d}-{month:02d}-01"


def date_to_year_month(date_str: str):
    """Преобразовать строку даты в (year, month)."""
    dt = pd.to_datetime(date_str)
    return dt.year, dt.month


if __name__ == "__main__":
    # Тестовый запуск
    print("Тест data_loader.py")
    print(f"Регион обучения: {TRAIN_BBOX}")
    print(f"Бассейн Или-Балхаш: {BASIN_BBOX}")
    print("Все функции загружены!")
