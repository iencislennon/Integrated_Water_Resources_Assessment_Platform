# 💧 Интегрированная платформа оценки водных ресурсов

**Регион:** Суб-бассейн Или → Балхаш  
**Назначение:** Демо-прототип для Oxford Saïd Global Climate Tech Challenge 2026, направление «Дефицит воды»

---

## 🎯 Что делает

Интерактивный веб-дашборд, показывающий «разукрупнённую» карту аномалий запасов воды в суб-бассейне Или-Балхаш с разрешением **500 м**.

- **GRACE/GRACE-FO** (~55 км) измеряет аномалии водных запасов
- **RandomForest** обучается предсказывать GRACE по мелким спутниковым признакам (осадки CHIRPS, эвапотранспирация MOD16, NDVI MOD13)
- Модель применяется к каждому пикселю 500 м
- **Валидация** по независимым данным уровня оз. Балхаш (DAHITI)

> ⚠️ **Честное ограничение:** GRACE измеряет бассейн целиком — реку Или, Капшагайское водохранилище и озеро Балхаш вместе. Разделить сигналы невозможно. Это физическое ограничение, не баг.

---

## 🚀 Быстрый старт (Happy Path)

### 1. Установка

```bash
git clone <repo-url>
cd ili-balkhash-water-platform
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Настройка доступов

#### Google Earth Engine
1. Создайте проект в [Google Cloud Console](https://console.cloud.google.com/)
2. Включите Earth Engine API
3. Зарегистрируйте как [некоммерческий](https://developers.google.com/earth-engine/guides/noncommercial_tiers)
4. Пройдите верификацию (обязательна с 26.09.2025)
5. Аутентификация:
   ```bash
   python -c "import ee; ee.Authenticate(); ee.Initialize(project='YOUR-PROJECT-ID')"
   ```

#### DAHITI (уровень оз. Балхаш)
1. Зарегистрируйтесь на [dahiti.dgfi.tum.de](https://dahiti.dgfi.tum.de/)
2. Получите персональный **API-ключ** в личном кабинете (это не логин/пароль от сайта — см. [FAQ](https://dahiti.dgfi.tum.de/en/frequently-asked-questions/api-key/))
3. ID оз. Балхаш: `91` (по умолчанию)

### 3. Запуск пайплайна

```bash
# Полный пайплайн (с демо-данными)
python run_pipeline.py --use-demo

# Или пошагово
python run_pipeline.py --step init --project-id YOUR_PROJECT_ID
python run_pipeline.py --step explore
python run_pipeline.py --step train
python run_pipeline.py --step predict --year 2023 --month 6
python run_pipeline.py --step validate --dahiti-api-key YOUR_DAHITI_API_KEY
python run_pipeline.py --step dashboard
```

### 4. Запуск дашборда

```bash
streamlit run dashboard/app.py
```

Откройте браузер: `http://localhost:8501`

---

## 📁 Структура проекта

```
ili-balkhash-water-platform/
├── README.md                    # Этот файл
├── requirements.txt             # Зависимости
├── run_pipeline.py              # Главный end-to-end скрипт
├── .streamlit/
│   └── config.toml              # Тема дашборда
├── data/
│   ├── raw/
│   │   └── dahiti_balkhash.json # Данные уровня воды
│   └── processed/
│       ├── training_data.csv    # Обучающая выборка
│       └── predictions_500m/    # GeoTIFF предсказаний
├── models/
│   └── rf_grace_downscaling.pkl # Обученная модель
├── src/
│   ├── data_loader.py           # Earth Engine + DAHITI API
│   ├── features.py              # Агрегация признаков
│   ├── model.py                 # RandomForest train + predict
│   └── validation.py            # Сравнение с DAHITI
├── dashboard/
│   └── app.py                   # Streamlit дашборд
└── assets/
    ├── validation_plot.html     # Интерактивный график
    ├── validation_report.md     # Отчёт валидации
    └── screenshots/             # Скриншоты для заявки
```

---

## 🛰️ Источники данных

| Данные | Earth Engine ID | Разрешение | Период |
|--------|----------------|------------|--------|
| GRACE/GRACE-FO Mascon | `NASA/GRACE/MASS_GRIDS_V04/MASCON` | ~55 км | 2002–н.в. |
| Осадки CHIRPS v3 | `UCSB-CHC/CHIRPS/V3/PENTAD` | ~5.5 км | 1981–н.в. |
| Эвапотранспирация MOD16 | `MODIS/061/MOD16A2GF` | 500 м | 2000–н.в. |
| NDVI MOD13 | `MODIS/061/MOD13Q1` | 250 м | 2000–н.в. |
| Граница бассейна | `WWF/HydroSHEDS/v1/Basins/hybas_6` | вектор | — |
| Уровень оз. Балхаш | DAHITI REST API v2 | точка | 2002–н.в. |

---

## 🔬 Методология

### Архитектура: 4 слоя

| Слой | Технология | Задача |
|------|-----------|--------|
| Данные | `earthengine-api` + `requests` | Загрузка, фильтрация, ресемплинг |
| Даунскейлинг | `scikit-learn` `RandomForestRegressor` | Обучение на широком регионе, предсказание 500 м |
| Валидация | `pandas` + `scipy.stats.pearsonr` | Сравнение с DAHITI |
| Визуализация | `streamlit` + `geemap` + `plotly` | Интерактивный дашборд |

### Пайплайн

1. **Подготовка** — сетка GRACE (0.5° × 0.5°) по региону Казахстан + ЦА
2. **Агрегация** — для каждой ячейки GRACE: средние осадки, ET, NDVI
3. **Обучение** — RandomForest: X = [precip, ET, NDVI, month_sin, month_cos], y = GRACE lwe_thickness
4. **Предсказание** — модель на каждый пиксель 500 м внутри бассейна
5. **Валидация** — сравнение усреднённых предсказаний с уровнем оз. Балхаш (DAHITI)

### Критерии приёмки

- ✅ R² ≥ 0.5 на test (временное разбиение: train 2002–2020, test 2021–2024)
- ✅ Корреляция с DAHITI ≥ 0.3 (стандартизированные ряды)

---

## 🖥️ Дашборд

### Компоненты

- **🗺️ Карта** — аномалии водных запасов 500 м, цветовая шкала RdBu
- **📈 Временной ряд** — GRACE даунскейлинг + уровень оз. Балхаш
- **🔗 Валидация** — наложение стандартизированных рядов + корреляция
- **🛰️ Доп. слои** — NDVI, ET, осадки (переключатель в сайдбаре)

### Деплой

```bash
# Локально
streamlit run dashboard/app.py

# На Streamlit Cloud
# 1. Push на GitHub
# 2. Подключить репозиторий на https://share.streamlit.io
# 3. Получить публичную ссылку для заявки
```

---

## ⚠️ Ограничения и честные оговорки

1. **GRACE видит бассейн целиком** — невозможно разделить сигналы реки Или, Капшагайского водохранилища и оз. Балхаш
2. **Капшагайское водохранилище** — регулируемый сток создаёт задержку между осадками и уровнем озера
3. **MOD16A2GF** — gap-filled продукт с годовой задержкой; для актуальных месяцев используется MOD16A2 (с пропусками)
4. **DAHITI vs GRACE** — разные физические величины (абсолютный уровень vs аномалия запасов), сравниваем динамику через стандартизацию
5. **Разрешение 500 м** — предсказания интерполированы; реальная точность ограничена разрешением входных признаков

---

## 📚 Ключевые ссылки

- [Earth Engine GRACE](https://developers.google.com/earth-engine/datasets/catalog/NASA_GRACE_MASS_GRIDS_V04_MASCON)
- [Earth Engine CHIRPS v3](https://developers.google.com/earth-engine/datasets/catalog/UCSB-CHC_CHIRPS_V3_PENTAD)
- [Earth Engine MOD16A2GF](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD16A2GF)
- [Earth Engine MOD13Q1](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD13Q1)
- [DAHITI](https://dahiti.dgfi.tum.de/)
- [geemap docs](https://geemap.org)
- [Streamlit sharing](https://share.streamlit.io)

---

## 📄 Лицензия

MIT License — для некоммерческого использования в рамках конкурса Oxford Saïd Global Climate Tech Challenge 2026.

---

*Разработано для Oxford Saïd Global Climate Tech Challenge 2026 | Направление: Дефицит воды*
