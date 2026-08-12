# 💧 Integrated Water Resources Assessment Platform

**Region:** Ili Sub-Basin → Lake Balkhash
**Purpose:** Demo prototype for the **Oxford Saïd Global Climate Tech Challenge 2026**
**Track:** Water Scarcity

**Languages:**
🇬🇧 [English](#english) · 🇨🇳 [中文](#中文) · 🇷🇺 [Русский](#русский)

---

<a name="english"></a>

# 🇬🇧 English

## 💧 Integrated Water Resources Assessment Platform

**Region:** Ili Sub-Basin → Lake Balkhash
**Purpose:** Demo prototype for the **Oxford Saïd Global Climate Tech Challenge 2026**
**Track:** Water Scarcity

---

## 🎯 What It Does

This project is an interactive web dashboard for monitoring and analyzing **water storage anomalies across the Ili–Balkhash sub-basin** at an effective spatial resolution of **500 m**.

The platform combines large-scale satellite observations with higher-resolution environmental indicators:

* **GRACE / GRACE-FO** (~55 km) measures large-scale terrestrial water storage anomalies.
* **Random Forest** learns the relationship between GRACE water-storage anomalies and higher-resolution satellite-derived features.
* The trained model is applied across the basin at **500 m resolution**.
* **DAHITI satellite altimetry** provides an independent validation signal using Lake Balkhash water-level observations.

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone <repo-url>
cd ili-balkhash-water-platform

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure Data Access

#### Google Earth Engine

1. Create a project in the [Google Cloud Console](https://console.cloud.google.com/).
2. Enable the Earth Engine API.
3. Register for the [Earth Engine noncommercial tier](https://developers.google.com/earth-engine/guides/noncommercial_tiers).
4. Complete the required verification.
5. Authenticate:

```bash
python -c "import ee; ee.Authenticate(); ee.Initialize(project='YOUR-PROJECT-ID')"
```

#### DAHITI

1. Register at [DAHITI](https://dahiti.dgfi.tum.de/).
2. Obtain a personal **API key** from your account.
3. The default Lake Balkhash dataset ID used by this project is:

```text
91
```

The DAHITI API key is separate from your website login credentials. See the [DAHITI API FAQ](https://dahiti.dgfi.tum.de/en/frequently-asked-questions/api-key/) for details.

---

## 3. Run the Pipeline

### Full pipeline with demo data

```bash
python run_pipeline.py --use-demo
```

### Or execute individual stages

```bash
python run_pipeline.py --step init --project-id YOUR_PROJECT_ID

python run_pipeline.py --step explore

python run_pipeline.py --step train

python run_pipeline.py --step predict --year 2023 --month 6

python run_pipeline.py --step validate \
    --dahiti-api-key YOUR_DAHITI_API_KEY

python run_pipeline.py --step dashboard
```

---

## 4. Launch the Dashboard

```bash
streamlit run dashboard/app.py
```

Open:

```text
http://localhost:8501
```

---

# 📁 Project Structure

```text
ili-balkhash-water-platform/
│
├── README.md
├── requirements.txt
├── run_pipeline.py
│
├── .streamlit/
│   └── config.toml
│
├── data/
│   ├── raw/
│   │   └── dahiti_balkhash.json
│   │
│   └── processed/
│       ├── training_data.csv
│       └── predictions_500m/
│
├── models/
│   └── rf_grace_downscaling.pkl
│
├── src/
│   ├── data_loader.py
│   ├── features.py
│   ├── model.py
│   └── validation.py
│
├── dashboard/
│   └── app.py
│
└── assets/
    ├── validation_plot.html
    ├── validation_report.md
    └── screenshots/
```

### Main Components

| File / Directory     | Purpose                               |
| -------------------- | ------------------------------------- |
| `run_pipeline.py`    | End-to-end pipeline entry point       |
| `src/data_loader.py` | Earth Engine and DAHITI data access   |
| `src/features.py`    | Feature extraction and aggregation    |
| `src/model.py`       | Random Forest training and prediction |
| `src/validation.py`  | Independent validation                |
| `dashboard/app.py`   | Streamlit dashboard                   |
| `data/processed/`    | Processed datasets and predictions    |
| `models/`            | Trained machine-learning model        |
| `assets/`            | Validation reports and screenshots    |

---

# 🛰️ Data Sources

| Dataset                      | Earth Engine ID / Source           |         Resolution | Period       |
| ---------------------------- | ---------------------------------- | -----------------: | ------------ |
| GRACE / GRACE-FO Mascon      | `NASA/GRACE/MASS_GRIDS_V04/MASCON` |             ~55 km | 2002–present |
| CHIRPS v3 Precipitation      | `UCSB-CHC/CHIRPS/V3/PENTAD`        |            ~5.5 km | 1981–present |
| MOD16A2GF Evapotranspiration | `MODIS/061/MOD16A2GF`              |              500 m | 2000–present |
| MOD13Q1 NDVI                 | `MODIS/061/MOD13Q1`                |              250 m | 2000–present |
| Basin Boundary               | `WWF/HydroSHEDS/v1/Basins/hybas_6` |             Vector | —            |
| Lake Balkhash Water Level    | DAHITI REST API v2                 | Point observations | 2002–present |

---

# 🔬 Methodology

## Architecture

The system consists of four main layers:

| Layer         | Technology                        | Role                                              |
| ------------- | --------------------------------- | ------------------------------------------------- |
| Data          | `earthengine-api` + `requests`    | Data acquisition, filtering and preprocessing     |
| Downscaling   | `scikit-learn` Random Forest      | Learn large-scale water-storage relationships     |
| Validation    | `pandas` + `scipy.stats.pearsonr` | Compare predictions with independent observations |
| Visualization | `Streamlit` + `geemap` + `Plotly` | Interactive analysis and visualization            |

---

## Pipeline

### 1. Data Preparation

A coarse GRACE grid covering Kazakhstan and Central Asia is generated.

### 2. Feature Aggregation

For every GRACE grid cell, satellite-derived predictors are aggregated:

* precipitation
* evapotranspiration
* NDVI
* seasonal variables

### 3. Model Training

A Random Forest regression model is trained using:

```text
X = [
    precipitation,
    evapotranspiration,
    NDVI,
    month_sin,
    month_cos
]

y = GRACE terrestrial water storage anomaly
```

The temporal split is:

```text
Training: 2002–2020
Testing:  2021–2024
```

### 4. Spatial Downscaling

The trained model is applied to the higher-resolution satellite features for every **500 m pixel inside the Ili–Balkhash basin**.

### 5. Independent Validation

The spatially averaged prediction is compared against an independent Lake Balkhash water-level time series obtained from DAHITI.

---

# 📊 Evaluation Criteria

The prototype targets the following acceptance criteria:

* **Test R² ≥ 0.50**
* **Correlation with DAHITI ≥ 0.30**
* Temporal holdout rather than random train/test splitting
* Independent validation using Lake Balkhash altimetry

---

# 🖥️ Dashboard

The dashboard provides four main analytical components.

### 🗺️ Water Storage Anomaly Map

Interactive visualization of estimated water-storage anomalies at **500 m resolution**.

### 📈 Temporal Analysis

Comparison of:

* GRACE observations
* downscaled water-storage estimates
* Lake Balkhash water-level observations

### 🔗 Independent Validation

Standardized time series and correlation analysis between the model output and DAHITI observations.

### 🛰️ Environmental Layers

Optional visualization of:

* NDVI
* evapotranspiration
* precipitation

---

# ☁️ Deployment

## Local

```bash
streamlit run dashboard/app.py
```

## Streamlit Cloud

1. Push the repository to GitHub.
2. Connect the repository to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Configure the required secrets and environment variables.
4. Deploy `dashboard/app.py`.
5. Use the resulting public URL for the competition submission.

---

# 📚 Key Resources

* [Google Earth Engine — GRACE Mass Grids](https://developers.google.com/earth-engine/datasets/catalog/NASA_GRACE_MASS_GRIDS_V04_MASCON)
* [Google Earth Engine — CHIRPS v3](https://developers.google.com/earth-engine/datasets/catalog/UCSB-CHC_CHIRPS_V3_PENTAD)
* [Google Earth Engine — MOD16A2GF](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD16A2GF)
* [Google Earth Engine — MOD13Q1](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD13Q1)
* [DAHITI](https://dahiti.dgfi.tum.de/)
* [DAHITI API FAQ](https://dahiti.dgfi.tum.de/en/frequently-asked-questions/api-key/)
* [geemap Documentation](https://geemap.org/)
* [Streamlit Community Cloud](https://share.streamlit.io/)

---

# 📄 License

MIT License.

Developed as a prototype for the **Oxford Saïd Global Climate Tech Challenge 2026 — Water Scarcity Track**.

---

<a name="中文"></a>

# 🇨🇳 中文

## 💧 综合水资源评估平台

**区域：** 伊犁河流域 → 巴尔喀什湖
**用途：** Oxford Saïd Global Climate Tech Challenge 2026 演示原型
**赛道：** 水资源短缺（Water Scarcity）

---

## 🎯 项目功能

本项目是一个交互式 Web 数据仪表板，用于分析和可视化**伊犁河—巴尔喀什湖子流域的水储量异常**，模型输出空间分辨率为 **500 米**。

平台将大尺度卫星观测与高分辨率环境遥感数据结合：

* **GRACE / GRACE-FO**（约 55 km）提供大尺度陆地水储量异常信息。
* **Random Forest（随机森林）**学习 GRACE 水储量异常与高分辨率卫星特征之间的关系。
* 训练完成后，模型应用于流域内部的 **500 m 像素**。
* 使用 **DAHITI 卫星测高数据**中的巴尔喀什湖水位作为独立验证数据。

---

## 🚀 快速开始

### 1. 安装

```bash
git clone <repo-url>
cd ili-balkhash-water-platform

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. 配置数据访问

#### Google Earth Engine

1. 在 [Google Cloud Console](https://console.cloud.google.com/) 创建项目。
2. 启用 Earth Engine API。
3. 注册 [Earth Engine 非商业使用计划](https://developers.google.com/earth-engine/guides/noncommercial_tiers)。
4. 完成所需验证。
5. 进行身份验证：

```bash
python -c "import ee; ee.Authenticate(); ee.Initialize(project='YOUR-PROJECT-ID')"
```

#### DAHITI

1. 在 [DAHITI](https://dahiti.dgfi.tum.de/) 注册账户。
2. 在账户中获取个人 **API Key**。
3. 默认使用的巴尔喀什湖数据集 ID：

```text
91
```

DAHITI API Key 与网站登录密码是不同的凭证。具体信息请参阅 [DAHITI API FAQ](https://dahiti.dgfi.tum.de/en/frequently-asked-questions/api-key/)。

---

## 3. 运行数据处理流程

### 使用 Demo 数据运行完整流程

```bash
python run_pipeline.py --use-demo
```

### 或逐步运行

```bash
python run_pipeline.py --step init --project-id YOUR_PROJECT_ID

python run_pipeline.py --step explore

python run_pipeline.py --step train

python run_pipeline.py --step predict --year 2023 --month 6

python run_pipeline.py --step validate \
    --dahiti-api-key YOUR_DAHITI_API_KEY

python run_pipeline.py --step dashboard
```

---

## 4. 启动仪表板

```bash
streamlit run dashboard/app.py
```

浏览器打开：

```text
http://localhost:8501
```

---

# 📁 项目结构

```text
ili-balkhash-water-platform/
│
├── README.md
├── requirements.txt
├── run_pipeline.py
│
├── .streamlit/
│   └── config.toml
│
├── data/
│   ├── raw/
│   │   └── dahiti_balkhash.json
│   │
│   └── processed/
│       ├── training_data.csv
│       └── predictions_500m/
│
├── models/
│   └── rf_grace_downscaling.pkl
│
├── src/
│   ├── data_loader.py
│   ├── features.py
│   ├── model.py
│   └── validation.py
│
├── dashboard/
│   └── app.py
│
└── assets/
    ├── validation_plot.html
    ├── validation_report.md
    └── screenshots/
```

---

# 🛰️ 数据源

| 数据                      | Earth Engine ID / 来源               |     分辨率 | 时间范围    |
| ----------------------- | ---------------------------------- | ------: | ------- |
| GRACE / GRACE-FO Mascon | `NASA/GRACE/MASS_GRIDS_V04/MASCON` |  ~55 km | 2002–至今 |
| CHIRPS v3 降水            | `UCSB-CHC/CHIRPS/V3/PENTAD`        | ~5.5 km | 1981–至今 |
| MOD16A2GF 蒸散发           | `MODIS/061/MOD16A2GF`              |   500 m | 2000–至今 |
| MOD13Q1 NDVI            | `MODIS/061/MOD13Q1`                |   250 m | 2000–至今 |
| 流域边界                    | `WWF/HydroSHEDS/v1/Basins/hybas_6` |      矢量 | —       |
| 巴尔喀什湖水位                 | DAHITI REST API v2                 |     点观测 | 2002–至今 |

---

# 🔬 方法论

## 系统架构

平台由四个主要层组成：

| 层级   | 技术                                | 任务                 |
| ---- | --------------------------------- | ------------------ |
| 数据层  | `earthengine-api` + `requests`    | 数据获取、过滤与预处理        |
| 降尺度层 | `scikit-learn` Random Forest      | 学习大尺度水储量与环境变量之间的关系 |
| 验证层  | `pandas` + `scipy.stats.pearsonr` | 与独立观测进行比较          |
| 可视化层 | `Streamlit` + `geemap` + `Plotly` | 交互式分析与可视化          |

---

## 数据处理流程

### 1. 数据准备

建立覆盖哈萨克斯坦及中亚区域的 GRACE 粗分辨率网格。

### 2. 特征聚合

对于每一个 GRACE 网格单元，计算：

* 降水
* 蒸散发
* NDVI
* 季节性特征

### 3. 模型训练

Random Forest 回归模型使用：

```text
X = [
    precipitation,
    evapotranspiration,
    NDVI,
    month_sin,
    month_cos
]

y = GRACE terrestrial water storage anomaly
```

时间划分：

```text
训练集：2002–2020
测试集：2021–2024
```

### 4. 空间降尺度

训练完成后，将模型应用于流域内部的高分辨率遥感特征，并生成 **500 m** 空间分辨率的估计结果。

### 5. 独立验证

将流域范围内的预测结果进行空间平均，并与 DAHITI 提供的巴尔喀什湖水位时间序列进行比较。

---

# 📊 评估标准

原型系统的目标指标：

* **测试集 R² ≥ 0.50**
* **与 DAHITI 的相关系数 ≥ 0.30**
* 使用时间序列划分，而不是随机划分训练集和测试集
* 使用独立的湖泊卫星测高数据进行验证

---

# 🖥️ 仪表板

### 🗺️ 水储量异常地图

展示 **500 m** 分辨率的估计水储量异常。

### 📈 时间序列

同时展示：

* GRACE 观测
* 降尺度后的水储量估计
* 巴尔喀什湖水位

### 🔗 独立验证

展示标准化时间序列以及模型输出与 DAHITI 数据之间的相关性。

### 🛰️ 环境图层

可以切换查看：

* NDVI
* 蒸散发
* 降水

---

# ☁️ 部署

## 本地部署

```bash
streamlit run dashboard/app.py
```

## Streamlit Cloud

1. 将代码推送到 GitHub。
2. 在 [Streamlit Community Cloud](https://share.streamlit.io/) 连接仓库。
3. 配置必要的 Secrets 和环境变量。
4. 将 `dashboard/app.py` 设置为入口。
5. 获取公开 URL，用于竞赛申请。

---

# 📚 关键资源

* [Google Earth Engine — GRACE Mass Grids](https://developers.google.com/earth-engine/datasets/catalog/NASA_GRACE_MASS_GRIDS_V04_MASCON)
* [Google Earth Engine — CHIRPS v3](https://developers.google.com/earth-engine/datasets/catalog/UCSB-CHC_CHIRPS_V3_PENTAD)
* [Google Earth Engine — MOD16A2GF](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD16A2GF)
* [Google Earth Engine — MOD13Q1](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD13Q1)
* [DAHITI](https://dahiti.dgfi.tum.de/)
* [DAHITI API FAQ](https://dahiti.dgfi.tum.de/en/frequently-asked-questions/api-key/)
* [geemap Documentation](https://geemap.org/)
* [Streamlit Community Cloud](https://share.streamlit.io/)

---

# 📄 许可证

MIT License。

本项目为 **Oxford Saïd Global Climate Tech Challenge 2026 — Water Scarcity Track** 开发的演示原型。

---

<a name="русский"></a>

# 🇷🇺 Русский

## 💧 Интегрированная платформа оценки водных ресурсов

**Регион:** суб-бассейн Или → озеро Балхаш
**Назначение:** демонстрационный прототип для **Oxford Saïd Global Climate Tech Challenge 2026**
**Направление:** дефицит воды (Water Scarcity)

---

## 🎯 Что делает проект

Проект представляет собой интерактивный веб-дашборд для анализа и визуализации **аномалий запасов воды в суб-бассейне Или–Балхаш** с пространственным разрешением выходных данных **500 м**.

Платформа объединяет крупномасштабные спутниковые наблюдения с более детальными экологическими показателями:

* **GRACE / GRACE-FO** (~55 км) измеряет крупномасштабные аномалии запасов воды на суше.
* **Random Forest** обучается находить связь между аномалиями GRACE и более детальными спутниковыми признаками.
* Обученная модель применяется к каждому **пикселю 500 м** внутри бассейна.
* **DAHITI** используется для независимой валидации через спутниковые измерения уровня озера Балхаш.

---

## 🚀 Быстрый старт

### 1. Установка

```bash
git clone <repo-url>
cd ili-balkhash-water-platform

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Настройка доступа к данным

#### Google Earth Engine

1. Создайте проект в [Google Cloud Console](https://console.cloud.google.com/).
2. Включите Earth Engine API.
3. Зарегистрируйтесь в [Earth Engine Noncommercial Tier](https://developers.google.com/earth-engine/guides/noncommercial_tiers).
4. Пройдите необходимую верификацию.
5. Выполните аутентификацию:

```bash
python -c "import ee; ee.Authenticate(); ee.Initialize(project='YOUR-PROJECT-ID')"
```

#### DAHITI

1. Зарегистрируйтесь на [DAHITI](https://dahiti.dgfi.tum.de/).
2. Получите персональный **API Key** в личном кабинете.
3. Используемый по умолчанию ID озера Балхаш:

```text
91
```

API Key DAHITI отличается от логина и пароля веб-сайта. Подробнее см. [DAHITI API FAQ](https://dahiti.dgfi.tum.de/en/frequently-asked-questions/api-key/).

---

## 3. Запуск пайплайна

### Полный пайплайн с demo-данными

```bash
python run_pipeline.py --use-demo
```

### Пошаговый запуск

```bash
python run_pipeline.py --step init --project-id YOUR_PROJECT_ID

python run_pipeline.py --step explore

python run_pipeline.py --step train

python run_pipeline.py --step predict --year 2023 --month 6

python run_pipeline.py --step validate \
    --dahiti-api-key YOUR_DAHITI_API_KEY

python run_pipeline.py --step dashboard
```

---

## 4. Запуск дашборда

```bash
streamlit run dashboard/app.py
```

Откройте:

```text
http://localhost:8501
```

---

# 📁 Структура проекта

```text
ili-balkhash-water-platform/
│
├── README.md
├── requirements.txt
├── run_pipeline.py
│
├── .streamlit/
│   └── config.toml
│
├── data/
│   ├── raw/
│   │   └── dahiti_balkhash.json
│   │
│   └── processed/
│       ├── training_data.csv
│       └── predictions_500m/
│
├── models/
│   └── rf_grace_downscaling.pkl
│
├── src/
│   ├── data_loader.py
│   ├── features.py
│   ├── model.py
│   └── validation.py
│
├── dashboard/
│   └── app.py
│
└── assets/
    ├── validation_plot.html
    ├── validation_report.md
    └── screenshots/
```

---

# 🛰️ Источники данных

| Данные                      | Earth Engine ID / источник         |          Разрешение | Период    |
| --------------------------- | ---------------------------------- | ------------------: | --------- |
| GRACE / GRACE-FO Mascon     | `NASA/GRACE/MASS_GRIDS_V04/MASCON` |              ~55 км | 2002–н.в. |
| Осадки CHIRPS v3            | `UCSB-CHC/CHIRPS/V3/PENTAD`        |             ~5.5 км | 1981–н.в. |
| Эвапотранспирация MOD16A2GF | `MODIS/061/MOD16A2GF`              |               500 м | 2000–н.в. |
| NDVI MOD13Q1                | `MODIS/061/MOD13Q1`                |               250 м | 2000–н.в. |
| Граница бассейна            | `WWF/HydroSHEDS/v1/Basins/hybas_6` |              Вектор | —         |
| Уровень озера Балхаш        | DAHITI REST API v2                 | Точечные наблюдения | 2002–н.в. |

---

# 🔬 Методология

## Архитектура

Система состоит из четырёх основных слоёв:

| Слой         | Технология                        | Задача                                                             |
| ------------ | --------------------------------- | ------------------------------------------------------------------ |
| Данные       | `earthengine-api` + `requests`    | Получение, фильтрация и подготовка данных                          |
| Даунскейлинг | `scikit-learn` Random Forest      | Моделирование пространственной структуры крупномасштабного сигнала |
| Валидация    | `pandas` + `scipy.stats.pearsonr` | Сравнение с независимыми наблюдениями                              |
| Визуализация | `Streamlit` + `geemap` + `Plotly` | Интерактивный анализ                                               |

---

## Пайплайн

### 1. Подготовка данных

Создаётся крупномасштабная сетка GRACE, покрывающая Казахстан и Центральную Азию.

### 2. Агрегация признаков

Для каждой ячейки GRACE рассчитываются:

* осадки;
* эвапотранспирация;
* NDVI;
* сезонные признаки.

### 3. Обучение модели

Random Forest обучается на:

```text
X = [
    precipitation,
    evapotranspiration,
    NDVI,
    month_sin,
    month_cos
]

y = GRACE terrestrial water storage anomaly
```

Временное разделение:

```text
Обучение: 2002–2020
Тест:     2021–2024
```

### 4. Пространственный даунскейлинг

После обучения модель применяется к высокодетальным спутниковым признакам внутри бассейна, формируя оценку с пространственным разрешением **500 м**.

### 5. Независимая валидация

Среднее значение предсказаний по территории бассейна сравнивается с временным рядом уровня озера Балхаш из DAHITI.

---

# 📊 Критерии оценки

Целевые показатели прототипа:

* **R² на тестовой выборке ≥ 0.50**
* **Корреляция с DAHITI ≥ 0.30**
* Временное разделение train/test вместо случайного разбиения
* Независимая валидация по спутниковой альтиметрии озера Балхаш

---

# 🖥️ Дашборд

### 🗺️ Карта аномалий запасов воды

Интерактивная карта оценённых аномалий запасов воды с разрешением **500 м**.

### 📈 Временной ряд

Сравнение:

* исходных данных GRACE;
* даунскейлинговых оценок;
* уровня озера Балхаш.

### 🔗 Независимая валидация

Стандартизированные временные ряды и корреляция между результатами модели и данными DAHITI.

### 🛰️ Дополнительные слои

Можно включать:

* NDVI;
* эвапотранспирацию;
* осадки.

---

# ☁️ Деплой

## Локально

```bash
streamlit run dashboard/app.py
```

## Streamlit Cloud

1. Загрузите репозиторий на GitHub.
2. Подключите его через [Streamlit Community Cloud](https://share.streamlit.io/).
3. Настройте необходимые Secrets и переменные окружения.
4. Укажите `dashboard/app.py` как entry point.
5. Используйте полученный публичный URL для заявки на конкурс.

---

# 📚 Ключевые ресурсы

* [Google Earth Engine — GRACE Mass Grids](https://developers.google.com/earth-engine/datasets/catalog/NASA_GRACE_MASS_GRIDS_V04_MASCON)
* [Google Earth Engine — CHIRPS v3](https://developers.google.com/earth-engine/datasets/catalog/UCSB-CHC_CHIRPS_V3_PENTAD)
* [Google Earth Engine — MOD16A2GF](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD16A2GF)
* [Google Earth Engine — MOD13Q1](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD13Q1)
* [DAHITI](https://dahiti.dgfi.tum.de/)
* [DAHITI API FAQ](https://dahiti.dgfi.tum.de/en/frequently-asked-questions/api-key/)
* [geemap Documentation](https://geemap.org/)
* [Streamlit Community Cloud](https://share.streamlit.io/)

---

# 📄 Лицензия

MIT License.

Разработано как демонстрационный прототип для **Oxford Saïd Global Climate Tech Challenge 2026 — Water Scarcity Track**.
