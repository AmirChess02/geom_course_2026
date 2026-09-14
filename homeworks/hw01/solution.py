from pathlib import Path
import numpy as np
import pandas as pd
from data_utils import (
    read_wells, read_layers, read_pumping_test, 
    print_table_info, DatasetInfo, save_results
)
def main():
    print("=== Выполнение полного решения домашней работы №1 ===\n")
    # --- Шаг 3 & 12. Чтение и проверка базовой информации ---
    wells = read_wells()
    layers = read_layers()
    pumping = read_pumping_test()
    print_table_info("Wells", wells)
    print_table_info("Layers", layers)
    print_table_info("Pumping Test", pumping)
    # Использование класса DatasetInfo (п. 12)
    info_wells = DatasetInfo("Wells", wells)
    info_layers = DatasetInfo("Layers", layers)
    info_pumping = DatasetInfo("Pumping Test", pumping)
    print("Информация через класс DatasetInfo:")
    print(info_wells.describe())
    print(info_layers.describe())
    print(info_pumping.describe(), "\n")
    # --- Шаг 4. Очистка и проверки ---
    wells_clean = wells.dropna(subset=["pressure_mpa"]).copy()
    assert (wells_clean["radius_m"] > 0).all()
    assert (layers["thickness_m"] > 0).all()
    assert layers["porosity_fraction"].between(0, 1).all()
    # --- Шаг 5. Векторизованные операции ---
    wells_clean["pressure_pa"] = wells_clean["pressure_mpa"] * 1_000_000
    wells_clean["pressure_difference_mpa"] = 12.0 - wells_clean["pressure_mpa"]
    wells_clean["relative_change_percent"] = (wells_clean["pressure_difference_mpa"] / 12.0) * 100.0
    # --- Шаг 6. Математические функции NumPy ---
    # Синус, косинус и проверка координат
    theta_rad = wells_clean["azimuth_deg"] * np.pi / 180.0
    x_check = wells_clean["radius_m"] * np.cos(theta_rad)
    y_check = wells_clean["radius_m"] * np.sin(theta_rad)
    assert np.allclose(x_check, wells_clean["x_m"], atol=0.02)
    assert np.allclose(y_check, wells_clean["y_m"], atol=0.02)
    print("Проверка координат через np.allclose успешно пройдена!")
    # Логарифм расстояния
    wells_clean["log_radius"] = np.log(wells_clean["radius_m"])
    # Экспонента для времени откачки
    decay = np.exp(-pumping["time_h"] / 36.0)
    print(f"Экспонента decay (первое значение ~1.0): {decay.iloc[0]:.4f}")
    # --- Шаг 7. Минимум, максимум, среднее ---
    stats_dict = {
        "pressure_mpa": [wells_clean["pressure_mpa"].min(), wells_clean["pressure_mpa"].max(), wells_clean["pressure_mpa"].mean()],
        "radius_m": [wells_clean["radius_m"].min(), wells_clean["radius_m"].max(), wells_clean["radius_m"].mean()],
        "thickness_m": [layers["thickness_m"].min(), layers["thickness_m"].max(), layers["thickness_m"].mean()],
        "porosity_fraction": [layers["porosity_fraction"].min(), layers["porosity_fraction"].max(), layers["porosity_fraction"].mean()]
    }
    summary_stats = pd.DataFrame(stats_dict, index=["min", "max", "mean"])
    print("\nСводная статистика:\n", summary_stats, "\n")
    # --- Шаг 8. Выбор элементов (индексы, срезы, маски) ---
    pressures = wells_clean["pressure_mpa"].to_numpy()
    print("Первое давление:", pressures[0], "| Последнее давление:", pressures[-1])
    print("Первые три давления:", pressures[:3])
    print("Каждое второе давление:", pressures[::2])
    mean_press = pressures.mean()
    mask_below_mean = pressures < mean_press
    print("Давления ниже среднего:", pressures[mask_below_mean])
    far_wells = wells_clean[wells_clean["radius_m"] > 100]
    print(f"Скважины дальше 100 м (строк): {len(far_wells)}")
    # --- Шаг 9. Двумерный массив ---
    pressure_matrix = pumping[["boundary_pressure_mpa", "well_pressure_mpa"]].to_numpy()
    print(f"\n2D массив: ndim={pressure_matrix.ndim}, shape={pressure_matrix.shape}, size={pressure_matrix.size}, dtype={pressure_matrix.dtype}")
    # --- Шаг 10. Трехмерный массив ---
    experiment_1 = pressure_matrix
    experiment_2 = pressure_matrix + 0.05
    pressure_cube = np.stack([experiment_1, experiment_2], axis=0)
    print(f"3D массив shape: {pressure_cube.shape}")
    print("Первая таблица (cube[0]):\n", pressure_cube[0])
    print("Первая строка первой таблицы (cube[0, 0]):", pressure_cube[0, 0])
    print("Столбец давления в скважине для обеих таблиц (cube[:, :, 1]):\n", pressure_cube[:, :, 1])
    # --- Шаг 11. Транспонирование и reshape ---
    pressure_transposed = pressure_matrix.T
    print("Форма после транспонирования (.T):", pressure_transposed.shape)
    flat = pressure_cube.reshape(-1)
    restored = flat.reshape(pressure_cube.shape)
    assert np.allclose(restored, pressure_cube)
    print("Проверка reshape и восстановления через np.allclose пройдена успешно!")
    # --- Шаг 13. Сохранение и обратная загрузка ---
    processed_dir, exports_dir = save_results(wells_clean, summary_stats, pressure_matrix, pressure_cube)
    # Обратная загрузка массивов для проверки
    matrix_loaded = np.load(processed_dir / "pressure_matrix.npy")
    cube_loaded = np.load(processed_dir / "pressure_cube.npz")["pressure_cube"]
    assert np.allclose(matrix_loaded, pressure_matrix)
    assert np.allclose(cube_loaded, pressure_cube)
    print(f"\nВсе файлы успешно сохранены в:\n - {processed_dir}\n - {exports_dir}")
    print("Обратная загрузка матриц и куба прошла проверку allclose успешно!")
if __name__ == "__main__":
    main()
