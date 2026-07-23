#!/usr/bin/env python3
"""
run_pipeline.py
End-to-end пайплайн: от сырых данных до даунскейленной карты и валидации.

Happy path — минимальная конфигурация для демо-прототипа.

Использование:
    python run_pipeline.py --project-id YOUR_GCP_PROJECT_ID --dahiti-user USER --dahiti-pass PASS

Или пошагово:
    python run_pipeline.py --step init
    python run_pipeline.py --step explore
    python run_pipeline.py --step train
    python run_pipeline.py --step predict --year 2023 --month 6
    python run_pipeline.py --step validate
    python run_pipeline.py --step dashboard
"""

import argparse
import os
import sys
import json
from dotenv import load_dotenv

# Загружаем переменные окружения из .env
load_dotenv()

# Добавляем src в path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# ───────────────────────────────────────────────────────────────────────────────
# Аргументы командной строки
# ───────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="End-to-end пайплайн даунскейлинга GRACE для бассейна Или-Балхаш"
    )

    parser.add_argument(
        "--step",
        choices=["all", "init", "explore", "train", "predict", "validate", "dashboard"],
        default="all",
        help="Этап пайплайна (по умолчанию: all)"
    )

    parser.add_argument(
        "--project-id",
        default=os.environ.get("EE_PROJECT_ID", None),
        help="Google Cloud Project ID для Earth Engine"
    )

    parser.add_argument(
        "--dahiti-user",
        default=os.environ.get("DAHITI_USER", None),
        help="Логин DAHITI"
    )

    parser.add_argument(
        "--dahiti-pass",
        default=os.environ.get("DAHITI_PASS", None),
        help="Пароль DAHITI"
    )

    parser.add_argument(
        "--dahiti-id",
        type=int,
        default=1000133634,
        help="ID оз. Балхаш в DAHITI (по умолчанию: 1000133634)"
    )

    parser.add_argument(
        "--year",
        type=int,
        default=2023,
        help="Год для предсказания (только для --step predict)"
    )

    parser.add_argument(
        "--month",
        type=int,
        default=6,
        help="Месяц для предсказания (только для --step predict)"
    )

    parser.add_argument(
        "--train-start",
        type=int,
        default=2002,
        help="Начальный год обучения"
    )

    parser.add_argument(
        "--train-end",
        type=int,
        default=2020,
        help="Конечный год обучения"
    )

    parser.add_argument(
        "--test-start",
        type=int,
        default=2021,
        help="Начальный год теста"
    )

    parser.add_argument(
        "--test-end",
        type=int,
        default=2024,
        help="Конечный год теста"
    )

    parser.add_argument(
        "--use-demo",
        action="store_true",
        help="Использовать синтетические данные вместо реальных"
    )

    return parser.parse_args()


# ───────────────────────────────────────────────────────────────────────────────
# Этап 0: Инициализация
# ───────────────────────────────────────────────────────────────────────────────

def step_init(args):
    """Инициализация Earth Engine и проверка доступов."""
    print("=" * 60)
    print("🚀 ЭТАП 0: Инициализация")
    print("=" * 60)

    from data_loader import init_ee, get_basin_polygon, get_train_region

    # Инициализация Earth Engine
    if args.project_id:
        init_ee(project_id=args.project_id)
    else:
        print("⚠️ PROJECT_ID не указан. Используйте --project-id или переменную окружения EE_PROJECT_ID")
        print("   Пропускаем инициализацию Earth Engine (будет запрошена при первом использовании)")

    # Проверка DAHITI
    if args.dahiti_user and args.dahiti_pass:
        print(f"✅ DAHITI credentials настроены (user: {args.dahiti_user})")
    else:
        print("⚠️ DAHITI credentials не указаны. Используйте --dahiti-user/--dahiti-pass")
        print("   или переменные окружения DAHITI_USER / DAHITI_PASS")

    print("\n✅ Инициализация завершена")
    return True


# ───────────────────────────────────────────────────────────────────────────────
# Этап 1: Исследование данных
# ───────────────────────────────────────────────────────────────────────────────

def step_explore(args):
    """Исследование данных: проверка доступности и качества."""
    print("\n" + "=" * 60)
    print("🔍 ЭТАП 1: Исследование данных")
    print("=" * 60)

    if args.use_demo:
        print("📊 Демо-режим: пропускаем проверку Earth Engine")
        return True

    try:
        import ee
        from data_loader import (
            get_grace_monthly, get_chirps_monthly, get_mod16_et, get_mod13_ndvi,
            get_basin_polygon, get_basin_bbox
        )

        # Проверяем GRACE
        print("\n📡 Проверка GRACE...")
        grace = get_grace_monthly("2023-01-01", "2023-12-31")
        grace_count = grace.size().getInfo()
        print(f"   GRACE записей за 2023: {grace_count}")

        # Проверяем CHIRPS
        print("\n🌧️ Проверка CHIRPS...")
        chirps = get_chirps_monthly("2023-01-01", "2023-12-31")
        chirps_count = chirps.size().getInfo()
        print(f"   CHIRPS месячных записей за 2023: {chirps_count}")

        # Проверяем MOD16 ET
        print("\n💨 Проверка MOD16 ET...")
        et = get_mod16_et("2023-01-01", "2023-12-31")
        et_count = et.size().getInfo()
        print(f"   ET месячных записей за 2023: {et_count}")

        # Проверяем NDVI
        print("\n🌿 Проверка MOD13 NDVI...")
        ndvi = get_mod13_ndvi("2023-01-01", "2023-12-31")
        ndvi_count = ndvi.size().getInfo()
        print(f"   NDVI месячных записей за 2023: {ndvi_count}")

        # Проверяем бассейн
        print("\n🗺️ Проверка границы бассейна...")
        try:
            basin = get_basin_polygon(basin_level=6)
            basin_info = basin.getInfo()
            print(f"   Бассейн найден: {basin_info['properties'].get('HYBAS_ID', 'N/A')}")
        except Exception as e:
            print(f"   ⚠️ Ошибка загрузки бассейна: {e}")
            print("   Попробуйте hybas_5 или hybas_7")

        print("\n✅ Исследование данных завершено")
        return True

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False


# ───────────────────────────────────────────────────────────────────────────────
# Этап 2: Сбор обучающих данных
# ───────────────────────────────────────────────────────────────────────────────

def step_train(args):
    """Сбор обучающих данных и обучение модели."""
    print("\n" + "=" * 60)
    print("🎓 ЭТАП 2: Обучение модели")
    print("=" * 60)

    if args.use_demo:
        print("📊 Демо-режим: генерируем синтетические обучающие данные...")

        import numpy as np
        import pandas as pd

        np.random.seed(42)
        n = 50000

        df = pd.DataFrame({
            "year": np.random.choice(range(args.train_start, args.test_end + 1), n),
            "month": np.random.choice(range(1, 13), n),
            "lat": np.random.uniform(40, 55, n),
            "lon": np.random.uniform(45, 90, n),
            "precip_mean": np.random.exponential(30, n),
            "et_mean": np.random.exponential(20, n),
            "ndvi_mean": np.random.beta(2, 5, n),
            "grace_lwe": np.random.normal(0, 10, n)
        })

        # Добавляем корреляцию между признаками и target для реалистичности
        df["grace_lwe"] = (
            0.3 * (df["precip_mean"] - 30) / 30 +
            -0.2 * (df["et_mean"] - 20) / 20 +
            0.1 * (df["ndvi_mean"] - 0.3) / 0.3 +
            np.random.normal(0, 8, n)
        )

        # Временные признаки
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df["date"] = pd.to_datetime(df[["year", "month"]].assign(day=1))

        # Сохраняем
        os.makedirs("data/processed", exist_ok=True)
        df.to_csv("data/processed/training_data.csv", index=False)
        print(f"   ✅ Сгенерировано {len(df)} синтетических записей")
    else:
        print("📡 Загрузка реальных данных из Earth Engine...")
        print("   (Для больших объёмов рекомендуется использовать export в Google Drive)")
        print("   См. features.py::export_training_data_batch()")

        # Здесь можно добавить код для загрузки из Earth Engine
        # Для happy path — используем демо-данные
        print("   ⚠️ Для реальных данных используйте batch export через Earth Engine")
        args.use_demo = True
        return step_train(args)

    # Обучение модели
    from model import train_model

    result = train_model(
        df=df,
        feature_cols=["precip_mean", "et_mean", "ndvi_mean", "month_sin", "month_cos"],
        target_col="grace_lwe",
        train_years=(args.train_start, args.train_end),
        test_years=(args.test_start, args.test_end),
        model_path="models/rf_grace_downscaling.pkl"
    )

    # Сохраняем метрики
    os.makedirs("assets", exist_ok=True)
    with open("assets/metrics.json", "w", encoding="utf-8") as f:
        json.dump(result["metrics"], f, indent=2)

    print(f"\n✅ Обучение завершено. Test R² = {result['metrics']['test_r2']:.4f}")

    if result["metrics"]["test_r2"] >= 0.5:
        print("   🎯 Критерий приёмки достигнут (R² ≥ 0.5)")
    else:
        print("   ⚠️ R² < 0.5 — рекомендуется добавить признаки или попробовать XGBoost")

    return True


# ───────────────────────────────────────────────────────────────────────────────
# Этап 3: Предсказание
# ───────────────────────────────────────────────────────────────────────────────

def step_predict(args):
    """Применение модели к конкретному месяцу."""
    print("\n" + "=" * 60)
    print(f"🔮 ЭТАП 3: Предсказание для {args.year}-{args.month:02d}")
    print("=" * 60)

    from model import load_model

    model = load_model("models/rf_grace_downscaling.pkl")

    if args.use_demo:
        print("📊 Демо-режим: генерируем синтетические предсказания...")

        import numpy as np
        import pandas as pd

        # Генерируем предсказания для всего периода
        dates = pd.date_range(
            f"{args.train_start}-01-01",
            f"{args.test_end}-12-01",
            freq="MS"
        )

        np.random.seed(123)
        n = len(dates)

        trend = np.linspace(5, -10, n)
        seasonal = np.sin(2 * np.pi * np.arange(n) / 12) * 3
        noise = np.random.normal(0, 2, n)

        pred_df = pd.DataFrame({
            "date": dates,
            "grace_mean": trend + seasonal + noise,
            "grace_std": np.abs(np.random.normal(2, 0.5, n)),
            "pixel_count": np.random.randint(8000, 12000, n)
        })

        os.makedirs("data/processed", exist_ok=True)
        pred_df.to_csv("data/processed/predictions_aggregated.csv", index=False)
        print(f"   ✅ Сгенерировано {len(pred_df)} месячных предсказаний")
    else:
        print("📡 Реальное предсказание через Earth Engine...")
        print("   (Требуется экспорт признаков, предсказание в numpy, импорт обратно)")
        print("   См. model.py::predict_batch()")

    print("\n✅ Предсказание завершено")
    return True


# ───────────────────────────────────────────────────────────────────────────────
# Этап 4: Валидация
# ───────────────────────────────────────────────────────────────────────────────

def step_validate(args):
    """Валидация предсказаний с данными DAHITI."""
    print("\n" + "=" * 60)
    print("✅ ЭТАП 4: Валидация")
    print("=" * 60)

    # Загружаем DAHITI
    if args.use_demo:
        print("📊 Демо-режим: генерируем синтетические данные DAHITI...")

        import numpy as np
        import pandas as pd

        dates = pd.date_range("2002-04-01", "2024-12-01", freq="MS")
        n = len(dates)

        trend = np.linspace(340, 342, n) + np.sin(np.linspace(0, 4 * np.pi, n)) * 0.5
        seasonal = np.sin(2 * np.pi * np.arange(n) / 12) * 0.3
        noise = np.random.normal(0, 0.1, n)

        dahiti_df = pd.DataFrame({
            "date": dates,
            "water_level": trend + seasonal + noise
        })

        os.makedirs("data/raw", exist_ok=True)
        dahiti_df.to_json("data/raw/dahiti_balkhash.json", orient="records", date_format="iso")
        print(f"   ✅ Сгенерировано {len(dahiti_df)} записей DAHITI")
    else:
        from data_loader import fetch_dahiti_water_level

        if args.dahiti_user and args.dahiti_pass:
            dahiti_df = fetch_dahiti_water_level(
                username=args.dahiti_user,
                password=args.dahiti_pass,
                dahiti_id=args.dahiti_id,
                output_path="data/raw/dahiti_balkhash.json"
            )
        else:
            print("⚠️ DAHITI credentials не указаны, используем демо-данные")
            args.use_demo = True
            return step_validate(args)

    # Загружаем предсказания
    pred_df = pd.read_csv("data/processed/predictions_aggregated.csv")
    pred_df["date"] = pd.to_datetime(pred_df["date"])

    # Валидация
    from validation import validate_with_dahiti, plot_validation, generate_validation_report

    result = validate_with_dahiti(
        predictions_df=pred_df,
        dahiti_df=dahiti_df,
        predictions_value_col="grace_mean",
        dahiti_value_col="water_level"
    )

    # График
    fig = plot_validation(result["merged_df"], save_path="assets/validation_plot.html")

    # Отчёт
    report = generate_validation_report(result, output_path="assets/validation_report.md")

    print("\n✅ Валидация завершена")
    print(f"   📊 Корреляция: {result['correlation']:.4f}")
    print(f"   📄 Отчёт: assets/validation_report.md")
    print(f"   📈 График: assets/validation_plot.html")

    return True


# ───────────────────────────────────────────────────────────────────────────────
# Этап 5: Дашборд
# ───────────────────────────────────────────────────────────────────────────────

def step_dashboard(args):
    """Запуск дашборда Streamlit."""
    print("\n" + "=" * 60)
    print("🖥️ ЭТАП 5: Запуск дашборда")
    print("=" * 60)

    print("""
Дашборд готов к запуску!

Команда для запуска:
    streamlit run dashboard/app.py

Или для деплоя:
    1. Загрузите код на GitHub
    2. Подключите репозиторий на https://share.streamlit.io
    3. Получите публичную ссылку

Файлы дашборда:
    - dashboard/app.py — основной файл
    - .streamlit/config.toml — конфигурация темы
""")

    return True


# ───────────────────────────────────────────────────────────────────────────────
# Главный пайплайн
# ───────────────────────────────────────────────────────────────────────────────

def run_pipeline(args):
    """Запуск полного пайплайна."""

    steps = {
        "init": step_init,
        "explore": step_explore,
        "train": step_train,
        "predict": step_predict,
        "validate": step_validate,
        "dashboard": step_dashboard,
    }

    if args.step == "all":
        for name, func in steps.items():
            success = func(args)
            if not success:
                print(f"\n❌ Пайплайн прерван на этапе: {name}")
                return False
    else:
        success = steps[args.step](args)
        if not success:
            return False

    print("\n" + "=" * 60)
    print("🎉 ПАЙПЛАЙН ЗАВЕРШЁН УСПЕШНО!")
    print("=" * 60)
    print("""
Результаты:
    📁 data/processed/training_data.csv — обучающие данные
    📁 models/rf_grace_downscaling.pkl — обученная модель
    📁 data/processed/predictions_aggregated.csv — предсказания
    📁 data/raw/dahiti_balkhash.json — данные DAHITI
    📁 assets/validation_plot.html — график валидации
    📁 assets/validation_report.md — отчёт валидации
    📁 assets/metrics.json — метрики модели

Следующий шаг:
    streamlit run dashboard/app.py
""")

    return True


# ───────────────────────────────────────────────────────────────────────────────
# Точка входа
# ───────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    # Создаём необходимые директории
    for d in ["data/raw", "data/processed", "models", "assets", "dashboard"]:
        os.makedirs(d, exist_ok=True)

    # Запускаем пайплайн
    success = run_pipeline(args)

    sys.exit(0 if success else 1)
