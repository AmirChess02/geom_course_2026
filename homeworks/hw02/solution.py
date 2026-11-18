from pathlib import Path
import pandas as pd
from plot_utils import plot_pressure_by_radius, plot_pressure_over_time
def main() -> None:
    root_dir = Path(__file__).resolve().parents[2]
    pumping_path = root_dir / "data" / "raw" / "hw01" / "pumping_test.txt"
    wells_path = root_dir / "data" / "processed" / "hw01" / "wells_clean.csv"
    figures_dir = root_dir / "figures" / "hw02"
    pumping_test = pd.read_csv(pumping_path, sep="\t")
    wells_clean = pd.read_csv(wells_path)
    wells_sorted = wells_clean.sort_values("radius_m")
    plot_pressure_over_time(pumping_test, figures_dir)
    plot_pressure_by_radius(wells_sorted, figures_dir)
    assert (figures_dir / "pressure_over_time.png").exists()
    assert (figures_dir / "pressure_over_time.svg").exists()
    assert (figures_dir / "pressure_by_radius.png").exists()
    assert (figures_dir / "pressure_by_radius.svg").exists()
    print("Все графики успешно построены и проверены!")
if __name__ == "__main__":
    main()
