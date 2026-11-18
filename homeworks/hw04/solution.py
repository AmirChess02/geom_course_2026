from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
def main() -> None:
    root_dir = Path(__file__).resolve().parents[2]
    # Путь к сохраненным данным из ДЗ №3
    viz_data_path = root_dir / "homeworks" / "hw03" / "viz_data.npz"
    assert viz_data_path.exists(), f"Файл не найден: {viz_data_path}"
    # 1. Загрузка и проверка результатов ДЗ №3 (с allow_pickle=True для строк)
    loaded_data = np.load(viz_data_path, allow_pickle=True)
    time_h = loaded_data["time_h"]
    well_pressure_mpa = loaded_data["well_pressure_mpa"]
    well_id = loaded_data["well_id"]
    radius_m = loaded_data["radius_m"]
    observation_pressure_mpa = loaded_data["observation_pressure_mpa"]
    assert time_h.shape == (13,)
    assert well_pressure_mpa.shape == (13,)
    assert well_id.shape == (7,)
    assert radius_m.shape == (7,)
    assert observation_pressure_mpa.shape == (7,)
    assert np.all(np.diff(time_h) > 0)
    assert np.all(np.diff(radius_m) > 0)
    assert (radius_m > 0).all()
    # 2. Создание учебного двумерного поля давления
    time_component = well_pressure_mpa - well_pressure_mpa[0]
    radius_component = observation_pressure_mpa - observation_pressure_mpa[0]
    pressure_field = (
        well_pressure_mpa[0]
        + time_component[:, np.newaxis]
        + radius_component[np.newaxis, :]
    )
    assert pressure_field.shape == (13, 7)
    # 3. Транспонирование для согласования с осями (X - время, Y - расстояние)
    field_for_plot = pressure_field.T
    assert field_for_plot.shape == (7, 13)
    # 4. Общие параметры цветовой шкалы и уровней
    vmin = pressure_field.min()
    vmax = pressure_field.max()
    levels = np.linspace(vmin, vmax, 9)
    fig_hw04_dir = root_dir / "figures" / "hw04"
    fig_hw04_dir.mkdir(parents=True, exist_ok=True)
    # 8. Сравнение (pressure_maps): Панель A (pcolormesh) и Панель B (contourf + contour)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    # Панель A: Тепловая карта через pcolormesh
    mesh = ax1.pcolormesh(
        time_h,
        radius_m,
        field_for_plot,
        shading="auto",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        edgecolors="k",
        linewidth=0.3,
    )
    ax1.set_yscale("log")
    ax1.set_xlabel("Время, ч")
    ax1.set_ylabel("Расстояние, м")
    ax1.set_title("A. Тепловая карта (pcolormesh)")
    cbar1 = fig.colorbar(mesh, ax=ax1)
    cbar1.set_label("Давление, МПа")
    # Панель B: Заливка по уровням с изолиниями
    filled = ax2.contourf(
        time_h,
        radius_m,
        field_for_plot,
        levels=levels,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    lines = ax2.contour(
        time_h,
        radius_m,
        field_for_plot,
        levels=levels,
        colors="black",
        linewidths=0.6,
    )
    ax2.clabel(lines, fmt="%.2f")
    ax2.set_yscale("log")
    ax2.set_xlabel("Время, ч")
    ax2.set_ylabel("Расстояние, м")
    ax2.set_title("B. Заливка уровней и изолинии")
    cbar2 = fig.colorbar(filled, ax=ax2)
    cbar2.set_label("Давление, МПа")
    fig.savefig(fig_hw04_dir / "pressure_maps.png", dpi=200, bbox_inches="tight")
    fig.savefig(fig_hw04_dir / "pressure_maps.svg", bbox_inches="tight")
    plt.close(fig)
    # 9. Построение только изолиний (pressure_isolines)
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    isolines = ax.contour(
        time_h,
        radius_m,
        field_for_plot,
        levels=levels,
        cmap="viridis",
        linewidths=1.2,
    )
    ax.clabel(isolines, fmt="%.2f")
    ax.set_yscale("log")
    ax.set_xlabel("Время, ч")
    ax.set_ylabel("Расстояние, м")
    ax.set_title("Изолинии давления")
    cbar = fig.colorbar(isolines, ax=ax)
    cbar.set_label("Давление, МПа")
    ax.grid(True, alpha=0.3)
    fig.savefig(fig_hw04_dir / "pressure_isolines.png", dpi=200, bbox_inches="tight")
    fig.savefig(fig_hw04_dir / "pressure_isolines.svg", bbox_inches="tight")
    plt.close(fig)
    # 10. Подготовка и сохранение данных для ДЗ №5 (.npz)
    field_data_path = root_dir / "homeworks" / "hw04" / "field_data.npz"
    np.savez(
        field_data_path,
        time_h=time_h,
        radius_m=radius_m,
        well_id=well_id,
        pressure_field=pressure_field,
        contour_levels=levels,
    )
    # Проверка загрузки созданного файла
    loaded_field_data = np.load(field_data_path, allow_pickle=True)
    for key in ["time_h", "radius_m", "well_id", "pressure_field", "contour_levels"]:
        assert key in loaded_field_data.files, f"Отсутствует ключ {key} в field_data.npz"
    print("Все графики ДЗ №4 успешно построены, а файл field_data.npz проверен!")
if __name__ == "__main__":
    main()
