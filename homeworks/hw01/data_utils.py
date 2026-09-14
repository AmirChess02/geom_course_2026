from pathlib import Path
import pandas as pd
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw" / "hw01"
PROCESSED_DIR = ROOT / "data" / "processed" / "hw01"
EXPORTS_DIR = ROOT / "exports" / "hw01"
# Чтение данных
def read_wells(path: Path = DATA_DIR / "wells.csv") -> pd.DataFrame:
    return pd.read_csv(path)
def read_layers(path: Path = DATA_DIR / "layers.xlsx") -> pd.DataFrame:
    return pd.read_excel(path)
def read_pumping_test(path: Path = DATA_DIR / "pumping_test.txt") -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")
# Вывод информации о таблице
def print_table_info(name: str, table: pd.DataFrame):
    print(f"--- {name} ---")
    print("Shape:", table.shape)
    print("Columns:", list(table.columns))
    print("Dtypes:\n", table.dtypes)
    print("Head:\n", table.head(), "\n")
# Класс DatasetInfo из п. 12
class DatasetInfo:
    def __init__(self, name: str, table: pd.DataFrame):
        self.name = name
        self.rows, self.columns = table.shape
    def describe(self) -> str:
        return f"{self.name}: {self.rows} строк, {self.columns} столбцов"
# Сохранение результатов (п. 13)
def save_results(wells_clean, summary_stats, pressure_matrix, pressure_cube):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    # 1. CSV
    wells_clean.to_csv(PROCESSED_DIR / "wells_clean.csv", index=False)
    # 2. Excel (минимум, максимум, среднее)
    summary_stats.to_excel(PROCESSED_DIR / "table_summary.xlsx", index=True)
    # 3. NPY и NPZ
    np.save(PROCESSED_DIR / "pressure_matrix.npy", pressure_matrix)
    np.savez(PROCESSED_DIR / "pressure_cube.npz", pressure_cube=pressure_cube)
    # 4. TXT с итогами
    txt_path = EXPORTS_DIR / "results.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=== Итоги домашней работы №1 ===\n")
        f.write(f"pressure_matrix shape: {pressure_matrix.shape}, ndim: {pressure_matrix.ndim}, size: {pressure_matrix.size}\n")
        f.write(f"pressure_cube shape: {pressure_cube.shape}, ndim: {pressure_cube.ndim}, size: {pressure_cube.size}\n")
        f.write("\nСтатистика (min, max, mean):\n")
        f.write(summary_stats.to_string())
    return PROCESSED_DIR, EXPORTS_DIR
