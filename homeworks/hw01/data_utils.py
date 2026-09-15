from pathlib import Path
import pandas as pd
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw" / "hw01"
PROCESSED_DIR = ROOT / "data" / "processed" / "hw01"
EXPORTS_DIR = ROOT / "exports" / "hw01"
def read_wells(path: Path = DATA_DIR / "wells.csv") -> pd.DataFrame:
    return pd.read_csv(path)
def read_layers(path: Path = DATA_DIR / "layers.xlsx") -> pd.DataFrame:
    return pd.read_excel(path)
def read_pumping_test(path: Path = DATA_DIR / "pumping_test.txt") -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")
def print_table_info(name: str, table: pd.DataFrame):
    print(f"--- {name} ---")
    print("Shape:", table.shape)
    print("Columns:", list(table.columns))
    print("Dtypes:\n", table.dtypes)
    print("Head:\n", table.head(), "\n")
class DatasetInfo:
    def __init__(self, name: str, table: pd.DataFrame):
        self.name = name
        self.rows, self.columns = table.shape
    def describe(self) -> str:
        return f"{self.name}: {self.rows} строк, {self.columns} столбцов"
def save_results(wells_clean, summary_stats, pressure_matrix, pressure_cube):
    # Создание выходных папок согласно пункту 13
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    # 1. CSV через Pandas
    wells_clean.to_csv(PROCESSED_DIR / "wells_clean.csv", index=False)
    # 2. Excel (содержит минимум, максимум и среднее)
    summary_stats.to_excel(PROCESSED_DIR / "table_summary.xlsx", index=True)
    # 3. Двумерный массив через numpy.save
    np.save(PROCESSED_DIR / "pressure_matrix.npy", pressure_matrix)
    # 4. Трехмерный массив через numpy.savez
    np.savez(PROCESSED_DIR / "pressure_cube.npz", pressure_cube=pressure_cube)
    # 5. TXT файл с формами массивов и статистикой
    txt_path = EXPORTS_DIR / "results.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=== Итоги домашней работы №1 ===\n\n")
        f.write(f"pressure_matrix shape: {pressure_matrix.shape}\n")
        f.write(f"pressure_cube shape: {pressure_cube.shape}\n\n")
        f.write("Сводная статистика (минимум, максимум, среднее):\n")
        f.write(summary_stats.to_string())
    # Обратная загрузка для проверки через np.allclose
    matrix_loaded = np.load(PROCESSED_DIR / "pressure_matrix.npy")
    cube_loaded = np.load(PROCESSED_DIR / "pressure_cube.npz")["pressure_cube"]
    assert np.allclose(matrix_loaded, pressure_matrix)
    assert np.allclose(cube_loaded, pressure_cube)
    return PROCESSED_DIR, EXPORTS_DIR
