from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import pandas as pd
def main() -> None:
    root_dir = Path(__file__).resolve().parents[2]
    # 1. Проверка результатов ДЗ №4
    field_data_path = root_dir / "homeworks" / "hw04" / "field_data.npz"
    pressure_maps_svg = root_dir / "figures" / "hw04" / "pressure_maps.svg"
    assert field_data_path.exists(), f"Файл не найден: {field_data_path}"
    assert pressure_maps_svg.exists(), f"Файл не найден: {pressure_maps_svg}"
    assert pressure_maps_svg.stat().st_size > 0, "Файл pressure_maps.svg пуст"
    loaded_data = np.load(field_data_path, allow_pickle=True)
    time_h = loaded_data["time_h"]
    radius_m = loaded_data["radius_m"]
    well_id = loaded_data["well_id"]
    pressure_field = loaded_data["pressure_field"]
    contour_levels = loaded_data["contour_levels"]
    assert time_h.shape == (13,)
    assert radius_m.shape == (7,)
    assert well_id.shape == (7,)
    assert pressure_field.shape == (13, 7)
    assert contour_levels.shape == (9,)
    assert np.all(np.diff(time_h) > 0)
    assert np.all(np.diff(radius_m) > 0)
    # 2. Создание длинной таблицы (Long table)
    time_grid, radius_grid = np.meshgrid(
        time_h,
        radius_m,
        indexing="ij",
    )
    time_index, radius_index = np.indices(pressure_field.shape)
    well_names_long = well_id[radius_index.reshape(-1)]
    long_table = pd.DataFrame(
        {
            "time_h": time_grid.reshape(-1),
            "radius_m": radius_grid.reshape(-1),
            "well_id": well_names_long,
            "pressure_mpa": pressure_field.reshape(-1),
        }
    )
    assert long_table.shape == (91, 4)
    assert long_table["time_h"].nunique() == 13
    assert long_table["radius_m"].nunique() == 7
    assert not long_table.isna().any().any()
    hw05_dir = root_dir / "homeworks" / "hw05"
    hw05_dir.mkdir(parents=True, exist_ok=True)
    long_csv_path = hw05_dir / "pressure_long.csv"
    long_table.to_csv(long_csv_path, index=False)
    # Проверка повторной загрузки CSV
    loaded_csv = pd.read_csv(long_csv_path)
    assert loaded_csv.shape == (91, 4)
    assert loaded_csv["time_h"].nunique() == 13
    assert loaded_csv["radius_m"].nunique() == 7
    assert not loaded_csv.isna().any().any()
    # 3. Создание анимации (FuncAnimation)
    fig_hw05_dir = root_dir / "figures" / "hw05"
    fig_hw05_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.set_xscale("log")
    margin = 0.1
    y_min = pressure_field.min() - margin
    y_max = pressure_field.max() + margin
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Расстояние, м")
    ax.set_ylabel("Давление, МПа")
    ax.grid(True, alpha=0.3)
    (line,) = ax.plot(radius_m, pressure_field[0], marker="o", color="#1f77b4", lw=2)
    title_text = ax.set_title(f"Время: {time_h[0]:.1f} ч")
    def update(frame_index):
        y_data = pressure_field[frame_index]
        line.set_ydata(y_data)
        title_text.set_text(f"Время: {time_h[frame_index]:.1f} ч")
        return line, title_text
    anim = FuncAnimation(fig, update, frames=len(time_h), interval=400, blit=True)
    anim_path = fig_hw05_dir / "pressure_animation.gif"
    anim.save(anim_path, writer=PillowWriter(fps=3))
    plt.close(fig)
    print("Длинная таблица и анимация успешно созданы и проверены!")
if __name__ == "__main__":
    main()
