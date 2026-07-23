"""
dashboard/app.py
Интерактивный дашборд для мониторинга водных ресурсов бассейна Или-Балхаш.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
import sys

# Добавляем src в path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# ───────────────────────────────────────────────────────────────────────────────
# Конфигурация страницы
# ───────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Водный баланс Или → Балхаш",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ───────────────────────────────────────────────────────────────────────────────
# Стили CSS
# ───────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #888;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1c1f26 0%, #2d333b 100%);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #30363d;
    }
    .limitation-box {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 4px;
        color: #856404;
    }
    .success-box {
        background: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 4px;
        color: #155724;
    }
</style>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────────────────────
# Загрузка данных (заглушки / локальные файлы)
# ───────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_dahiti_data():
    """Загрузить данные DAHITI из локального JSON."""
    path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "dahiti_balkhash.json")
    if os.path.exists(path):
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
    return None


@st.cache_data
def load_training_data():
    """Загрузить обучающие данные."""
    path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "training_data.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df[["year", "month"]].assign(day=1))
        return df
    return None


@st.cache_data
def load_predictions():
    """Загрузить предсказанные данные."""
    path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "predictions_aggregated.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        return df
    return None


@st.cache_data
def load_validation():
    """Загрузить результаты валидации."""
    path = os.path.join(os.path.dirname(__file__), "..", "assets", "validation_plot.html")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return None


# ───────────────────────────────────────────────────────────────────────────────
# Генерация синтетических данных для демо (если реальных нет)
# ───────────────────────────────────────────────────────────────────────────────

def generate_demo_data():
    """Сгенерировать синтетические данные для демонстрации дашборда."""
    np.random.seed(42)

    # DAHITI: уровень оз. Балхаш (синтетический)
    dates = pd.date_range("2002-04-01", "2024-12-01", freq="MS")
    n = len(dates)

    # Базовый тренд + сезонность + шум
    trend = np.linspace(340, 342, n) + np.sin(np.linspace(0, 4 * np.pi, n)) * 0.5
    seasonal = np.sin(2 * np.pi * np.arange(n) / 12) * 0.3
    noise = np.random.normal(0, 0.1, n)
    water_level = trend + seasonal + noise

    dahiti_df = pd.DataFrame({
        "date": dates,
        "water_level": water_level
    })

    # GRACE даунскейлинг (синтетический, коррелированный с DAHITI)
    grace_trend = np.linspace(5, -10, n) + np.sin(np.linspace(0, 4 * np.pi, n)) * 3
    grace_seasonal = np.sin(2 * np.pi * np.arange(n) / 12) * 2
    grace_noise = np.random.normal(0, 1, n)
    grace_mean = grace_trend + grace_seasonal + grace_noise

    predictions_df = pd.DataFrame({
        "date": dates,
        "grace_mean": grace_mean,
        "grace_std": np.abs(np.random.normal(2, 0.5, n)),
        "pixel_count": np.random.randint(8000, 12000, n)
    })

    # Обучающие данные (синтетические)
    n_train = 50000
    train_df = pd.DataFrame({
        "year": np.random.choice(range(2002, 2025), n_train),
        "month": np.random.choice(range(1, 13), n_train),
        "lat": np.random.uniform(40, 55, n_train),
        "lon": np.random.uniform(45, 90, n_train),
        "precip_mean": np.random.exponential(30, n_train),
        "et_mean": np.random.exponential(20, n_train),
        "ndvi_mean": np.random.beta(2, 5, n_train),
        "grace_lwe": np.random.normal(0, 10, n_train)
    })
    train_df["date"] = pd.to_datetime(train_df[["year", "month"]].assign(day=1))

    return dahiti_df, predictions_df, train_df


# ───────────────────────────────────────────────────────────────────────────────
# Главная страница
# ───────────────────────────────────────────────────────────────────────────────

def main():
    # Заголовок
    st.markdown('<div class="main-header">💧 Водный баланс бассейна Или → Балхаш</div>', 
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Интегрированная платформа оценки водных ресурсов | '
                'Демо-прототип для Oxford Saïd Global Climate Tech Challenge 2026</div>',
                unsafe_allow_html=True)

    # ───────────────────────────────────────────────────────────────────────────
    # Боковая панель
    # ───────────────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Настройки")

        # Переключатель данных
        data_source = st.radio(
            "Источник данных:",
            ["Синтетические (демо)", "Реальные (Earth Engine)"],
            index=0
        )

        use_demo = (data_source == "Синтетические (демо)")

        # Диапазон дат
        st.subheader("📅 Период")
        date_range = st.slider(
            "Диапазон дат",
            min_value=2002,
            max_value=2024,
            value=(2002, 2024)
        )

        # Дополнительные слои
        st.subheader("🗺️ Слои карты")
        show_ndvi = st.checkbox("NDVI", value=False)
        show_et = st.checkbox("Эвапотранспирация", value=False)
        show_precip = st.checkbox("Осадки", value=False)

        # Описание метода
        st.subheader("ℹ️ О методе")
        with st.expander("Как это работает?"):
            st.markdown("""
            **Метод даунскейлинга GRACE:**

            1. **GRACE/GRACE-FO** (~55 км) измеряет аномалии водных запасов
            2. **RandomForest** обучается предсказывать GRACE по мелким признакам:
               - Осадки CHIRPS (~5.5 км)
               - Эвапотранспирация MOD16 (~500 м)
               - NDVI MOD13 (~250 м)
            3. Модель применяется к каждому пикселю 500 м
            4. **Валидация** по уровню оз. Балхаш (DAHITI)

            **Разрешение выхода:** 500 м
            **Регион обучения:** Казахстан + Центральная Азия
            """)

        with st.expander("⚠️ Ограничения"):
            st.markdown("""
            - GRACE видит **весь бассейн целиком** — нельзя разделить сигналы реки Или, 
              Капшагайского водохранилища и оз. Балхаш
            - **Капшагай** — регулируемый сток, создаёт задержку
            - MOD16A2GF имеет **годовую задержку**
            - Сравнение с DAHITI — **динамика**, не абсолютные величины
            """)

    # ───────────────────────────────────────────────────────────────────────────
    # Загрузка данных
    # ───────────────────────────────────────────────────────────────────────────
    if use_demo:
        dahiti_df, predictions_df, train_df = generate_demo_data()
        st.sidebar.success("✅ Загружены демо-данные")
    else:
        dahiti_df = load_dahiti_data()
        predictions_df = load_predictions()
        train_df = load_training_data()

        if dahiti_df is None or predictions_df is None:
            st.warning("⚠️ Реальные данные не найдены. Переключаюсь на демо-режим.")
            dahiti_df, predictions_df, train_df = generate_demo_data()

    # Фильтрация по дате
    mask_dahiti = (dahiti_df["date"].dt.year >= date_range[0]) & (dahiti_df["date"].dt.year <= date_range[1])
    mask_pred = (predictions_df["date"].dt.year >= date_range[0]) & (predictions_df["date"].dt.year <= date_range[1])

    dahiti_filtered = dahiti_df[mask_dahiti].copy()
    pred_filtered = predictions_df[mask_pred].copy()

    # ───────────────────────────────────────────────────────────────────────────
    # Метрики
    # ───────────────────────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="📊 Период данных",
            value=f"{date_range[0]}—{date_range[1]}"
        )

    with col2:
        latest_grace = pred_filtered["grace_mean"].iloc[-1] if len(pred_filtered) > 0 else 0
        st.metric(
            label="💧 Аномалия запасов (последний месяц)",
            value=f"{latest_grace:.1f} см",
            delta=f"{latest_grace - pred_filtered['grace_mean'].iloc[-2]:.1f} см" if len(pred_filtered) > 1 else None
        )

    with col3:
        latest_level = dahiti_filtered["water_level"].iloc[-1] if len(dahiti_filtered) > 0 else 0
        st.metric(
            label="🌊 Уровень оз. Балхаш",
            value=f"{latest_level:.1f} м",
            delta=f"{latest_level - dahiti_filtered['water_level'].iloc[-2]:.2f} м" if len(dahiti_filtered) > 1 else None
        )

    with col4:
        # Корреляция (если есть данные)
        if len(pred_filtered) > 0 and len(dahiti_filtered) > 0:
            merged = pd.merge_asof(
                pred_filtered.sort_values("date"),
                dahiti_filtered.sort_values("date"),
                on="date",
                direction="nearest"
            ).dropna()

            if len(merged) > 10:
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                grace_std = scaler.fit_transform(merged[["grace_mean"]]).flatten()
                dahiti_std = scaler.fit_transform(merged[["water_level"]]).flatten()
                corr = np.corrcoef(grace_std, dahiti_std)[0, 1]

                st.metric(
                    label="🔗 Корреляция с DAHITI",
                    value=f"{corr:.3f}"
                )
            else:
                st.metric(label="🔗 Корреляция с DAHITI", value="N/A")
        else:
            st.metric(label="🔗 Корреляция с DAHITI", value="N/A")

    # ───────────────────────────────────────────────────────────────────────────
    # Карта (заглушка / geemap)
    # ───────────────────────────────────────────────────────────────────────────
    st.header("🗺️ Карта аномалий водных запасов")

    try:
        import geemap.foliumap as geemap

        m = geemap.Map(center=[46.5, 78.0], zoom=7)
        m.add_basemap("HYBRID")

        # Добавляем полигон бассейна (упрощённый)
        basin_coords = [
            [74.0, 43.0], [82.0, 43.0], [82.0, 48.0], [74.0, 48.0], [74.0, 43.0]
        ]
        m.add_polygon(
            coords=basin_coords,
            name="Бассейн Или-Балхаш",
            color="black",
            fill_color=None,
            weight=2
        )

        # Маркер оз. Балхаш
        m.add_marker(
            location=[46.5, 78.0],
            popup="Оз. Балхаш",
            icon="info-sign"
        )

        # Если есть реальные данные — добавляем слой
        # (в демо-режиме показываем заглушку)

        m.to_streamlit(height=500)

    except Exception as e:
        st.info("🗺️ Интерактивная карта требует geemap. Установите: `pip install geemap[streamlit]`")

        # Fallback: статичная карта через plotly
        fig_map = go.Figure()

        # Контур бассейна
        fig_map.add_trace(go.Scattermapbox(
            mode="lines",
            lon=[74, 82, 82, 74, 74],
            lat=[43, 43, 48, 48, 43],
            line=dict(color="black", width=2),
            name="Бассейн Или-Балхаш"
        ))

        # Оз. Балхаш
        fig_map.add_trace(go.Scattermapbox(
            mode="markers",
            lon=[78.0],
            lat=[46.5],
            marker=dict(size=15, color="blue"),
            name="Оз. Балхаш",
            text=["Оз. Балхаш"],
            hovertemplate="%{text}<extra></extra>"
        ))

        fig_map.update_layout(
            mapbox_style="carto-darkmatter",
            mapbox_center={"lat": 46.5, "lon": 78.0},
            mapbox_zoom=6,
            height=500,
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=True
        )

        st.plotly_chart(fig_map, use_container_width=True)

    # ───────────────────────────────────────────────────────────────────────────
    # Временной ряд
    # ───────────────────────────────────────────────────────────────────────────
    st.header("📈 Динамика водных ресурсов")

    fig_ts = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            "Аномалия запасов воды (GRACE даунскейлинг, 500 м)",
            "Уровень оз. Балхаш (DAHITI)"
        ),
        vertical_spacing=0.12,
        shared_xaxes=True
    )

    # GRACE
    fig_ts.add_trace(
        go.Scatter(
            x=pred_filtered["date"],
            y=pred_filtered["grace_mean"],
            mode="lines",
            name="GRACE даунскейлинг",
            line=dict(color="#1f77b4", width=2),
            fill="tozeroy",
            fillcolor="rgba(31, 119, 180, 0.1)"
        ),
        row=1, col=1
    )

    # Доверительный интервал
    if "grace_std" in pred_filtered.columns:
        fig_ts.add_trace(
            go.Scatter(
                x=pred_filtered["date"],
                y=pred_filtered["grace_mean"] + pred_filtered["grace_std"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip"
            ),
            row=1, col=1
        )
        fig_ts.add_trace(
            go.Scatter(
                x=pred_filtered["date"],
                y=pred_filtered["grace_mean"] - pred_filtered["grace_std"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(31, 119, 180, 0.2)",
                name="±1 std",
                hoverinfo="skip"
            ),
            row=1, col=1
        )

    # DAHITI
    fig_ts.add_trace(
        go.Scatter(
            x=dahiti_filtered["date"],
            y=dahiti_filtered["water_level"],
            mode="lines",
            name="Уровень оз. Балхаш",
            line=dict(color="#ff7f0e", width=2),
            fill="tozeroy",
            fillcolor="rgba(255, 127, 14, 0.1)"
        ),
        row=2, col=1
    )

    fig_ts.update_layout(
        height=600,
        template="plotly_dark",
        hovermode="x unified",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    fig_ts.update_yaxes(title_text="Аномалия, см", row=1, col=1)
    fig_ts.update_yaxes(title_text="Уровень, м", row=2, col=1)
    fig_ts.update_xaxes(title_text="Год", row=2, col=1)

    st.plotly_chart(fig_ts, use_container_width=True)

    # ───────────────────────────────────────────────────────────────────────────
    # Сравнение (стандартизированные ряды)
    # ───────────────────────────────────────────────────────────────────────────
    st.header("🔗 Валидация: сравнение с DAHITI")

    if len(pred_filtered) > 0 and len(dahiti_filtered) > 0:
        merged = pd.merge_asof(
            pred_filtered.sort_values("date"),
            dahiti_filtered.sort_values("date"),
            on="date",
            direction="nearest"
        ).dropna()

        if len(merged) > 10:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()

            merged["grace_std"] = scaler.fit_transform(merged[["grace_mean"]]).flatten()
            merged["dahiti_std"] = scaler.fit_transform(merged[["water_level"]]).flatten()

            fig_val = go.Figure()

            fig_val.add_trace(go.Scatter(
                x=merged["date"],
                y=merged["grace_std"],
                mode="lines+markers",
                name="GRACE даунскейлинг (Z-score)",
                line=dict(color="#1f77b4", width=2),
                marker=dict(size=4)
            ))

            fig_val.add_trace(go.Scatter(
                x=merged["date"],
                y=merged["dahiti_std"],
                mode="lines+markers",
                name="DAHITI (Z-score)",
                line=dict(color="#ff7f0e", width=2),
                marker=dict(size=4)
            ))

            corr = np.corrcoef(merged["grace_std"], merged["dahiti_std"])[0, 1]

            fig_val.add_annotation(
                x=0.02, y=0.98,
                xref="paper", yref="paper",
                text=f"r = {corr:.3f}",
                showarrow=False,
                font=dict(size=18, color="green"),
                bgcolor="rgba(0,0,0,0.7)",
                bordercolor="green",
                borderwidth=2
            )

            fig_val.update_layout(
                height=400,
                template="plotly_dark",
                yaxis_title="Z-score (стандартизированное значение)",
                xaxis_title="Год",
                hovermode="x unified"
            )

            st.plotly_chart(fig_val, use_container_width=True)

            # Интерпретация
            if corr >= 0.7:
                st.markdown('<div class="success-box">✅ <b>Отличное согласие</b> (r ≥ 0.7) — '
                           'метод надёжно улавливает динамику водных запасов.</div>',
                           unsafe_allow_html=True)
            elif corr >= 0.5:
                st.markdown('<div class="success-box">✅ <b>Хорошее согласие</b> (r = 0.5–0.7) — '
                           'основные тренды улавливаются корректно.</div>',
                           unsafe_allow_html=True)
            elif corr >= 0.3:
                st.markdown('<div class="limitation-box">⚠️ <b>Приемлемое согласие</b> (r = 0.3–0.5) — '
                           'для демо-прототипа приемлемо, но требует доработки.</div>',
                           unsafe_allow_html=True)
            else:
                st.markdown('<div class="limitation-box">⚠️ <b>Низкое согласие</b> (r < 0.3) — '
                           'метод требует доработки (больше признаков, другая модель).</div>',
                           unsafe_allow_html=True)
        else:
            st.info("Недостаточно данных для валидации")
    else:
        st.info("Загрузите данные DAHITI и предсказания для валидации")

    # ───────────────────────────────────────────────────────────────────────────
    # Дополнительные слои
    # ───────────────────────────────────────────────────────────────────────────
    if show_ndvi or show_et or show_precip:
        st.header("🛰️ Дополнительные спутниковые данные")

        cols = st.columns(sum([show_ndvi, show_et, show_precip]))
        idx = 0

        if show_ndvi:
            with cols[idx]:
                st.subheader("🌿 NDVI")
                # Синтетический NDVI
                ndvi_dates = pd.date_range("2020-01-01", "2020-12-01", freq="MS")
                ndvi_values = 0.2 + 0.3 * np.sin(2 * np.pi * np.arange(12) / 12 - np.pi/2)
                fig_ndvi = px.line(
                    x=ndvi_dates, y=ndvi_values,
                    labels={"x": "Месяц", "y": "NDVI"},
                    template="plotly_dark"
                )
                fig_ndvi.update_traces(line_color="green")
                st.plotly_chart(fig_ndvi, use_container_width=True)
            idx += 1

        if show_et:
            with cols[idx]:
                st.subheader("💨 Эвапотранспирация")
                et_dates = pd.date_range("2020-01-01", "2020-12-01", freq="MS")
                et_values = 20 + 40 * np.sin(2 * np.pi * np.arange(12) / 12 - np.pi/2)
                fig_et = px.line(
                    x=et_dates, y=et_values,
                    labels={"x": "Месяц", "y": "ET, мм/мес"},
                    template="plotly_dark"
                )
                fig_et.update_traces(line_color="cyan")
                st.plotly_chart(fig_et, use_container_width=True)
            idx += 1

        if show_precip:
            with cols[idx]:
                st.subheader("🌧️ Осадки")
                precip_dates = pd.date_range("2020-01-01", "2020-12-01", freq="MS")
                precip_values = 15 + 25 * np.sin(2 * np.pi * np.arange(12) / 12 - np.pi/3) + np.random.normal(0, 5, 12)
                fig_precip = px.bar(
                    x=precip_dates, y=np.maximum(precip_values, 0),
                    labels={"x": "Месяц", "y": "Осадки, мм/мес"},
                    template="plotly_dark"
                )
                fig_precip.update_traces(marker_color="lightblue")
                st.plotly_chart(fig_precip, use_container_width=True)

    # ───────────────────────────────────────────────────────────────────────────
    # Футер
    # ───────────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<center><small>"
        "Интегрированная платформа оценки водных ресурсов | "
        "Oxford Saïd Global Climate Tech Challenge 2026 | "
        "Данные: NASA GRACE, CHIRPS, MODIS, DAHITI | "
        "<a href='https://github.com/'>GitHub</a>"
        "</small></center>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
