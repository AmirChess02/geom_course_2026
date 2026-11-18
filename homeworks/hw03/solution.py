from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
def main() -> None:
    root_dir = Path(__file__).resolve().parents[2]
    hw02_fig_dir = root_dir / "figures" / "hw02"
    pumping_path = root_dir / "data" / "raw" / "hw01" / "pumping_test.txt"
    wells_path = root_dir / "data" / "processed" / "hw01" / "wells_clean.csv"
    required_figures = [
        hw02_fig_dir / "pressure_over_time.svg",
        hw02_fig_dir / "pressure_by_radius.svg",
    ]
    for path in required_figures:
        assert path.exists(), f"Файл не найден: {path}"
        assert path.stat().st_size > 0, f"Файл пуст: {path}"
    pumping_test = pd.read_csv(pumping_path, sep="\t")
    wells = pd.read_csv(wells_path)
    assert pumping_test.shape == (13, 4), "Неверная форма pumping_test"
    assert wells.shape[0] == 7, "Неверное количество скважин в wells"
    assert not pumping_test.isna().any().any(), "Найдены NaN в pumping_test"
    assert not wells.isna().any().any(), "Найдены NaN в wells"
    time_h = pumping_test["time_h"].to_numpy()
    boundary_pressure_mpa = pumping_test["boundary_pressure_mpa"].to_numpy()
    well_pressure_mpa = pumping_test["well_pressure_mpa"].to_numpy()
    well_id = wells["well_id"].to_numpy()
    radius_m = wells["radius_m"].to_numpy()
    observation_pressure_mpa = wells["pressure_mpa"].to_numpy()
    order = np.argsort(radius_m)
    well_id = well_id[order]
    radius_m = radius_m[order]
    observation_pressure_mpa = observation_pressure_mpa[order]
    pressure_drop_mpa = boundary_pressure_mpa - well_pressure_mpa
    observation_drop_mpa = 12.0 - observation_pressure_mpa
    fig_hw03_dir = root_dir / "figures" / "hw03"
    fig_hw03_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 8),
        constrained_layout=True,
    )
    # Панель A
    ax_a = axes[0, 0]
    ax_a.plot(time_h, boundary_pressure_mpa, label="Граница", marker="o", color="#1f77b4")
    ax_a.plot(time_h, well_pressure_mpa, label="Скважина", marker="s", color="#ff7f0e")
    ax_a.set_xlabel("Время, ч")
    ax_a.set_ylabel("Давление, МПа")
    ax_a.set_title("A. Временные ряды давления")
    ax_a.legend()
    ax_a.grid(True, alpha=0.3)
    # Панель B
    ax_b = axes[0, 1]
    sc = ax_b.scatter(
        radius_m,
        observation_pressure_mpa,
        c=observation_pressure_mpa,
        cmap="viridis",
        s=60,
        edgecolors="k",
    )
    ax_b.set_xlabel("Расстояние, м")
    ax_b.set_ylabel("Давление, МПа")
    ax_b.set_title("B. Наблюдательные скважины")
    ax_b.grid(True, alpha=0.3)
    cbar = fig.colorbar(sc, ax=ax_b)
    cbar.set_label("Давление, МПа")
    # Панель C
    ax_c = axes[1, 0]
    ax_c.bar(well_id, observation_drop_mpa, color="#2ca02c", alpha=0.8, edgecolor="k")
    ax_c.set_xlabel("ID скважины")
    ax_c.set_ylabel("Падение давления, МПа")
    ax_c.set_title("C. Снижение давления по скважинам")
    ax_c.grid(True, axis="y", alpha=0.3)
    # Панель D (boxplot)
    ax_d = axes[1, 1]
    ax_d.boxplot(
        [boundary_pressure_mpa, well_pressure_mpa],
        tick_labels=["Граница", "Скважина"],
        patch_artist=True,
        boxprops=dict(facecolor="#aec7e8", color="#1f77b4"),
        medianprops=dict(color="red", linewidth=1.5),
    )
    ax_d.set_ylabel("Давление, МПа")
    ax_d.set_title("D. Распределение давления")
    ax_d.grid(True, axis="y", alpha=0.3)
    fig.savefig(fig_hw03_dir / "pressure_overview.png", dpi=200, bbox_inches="tight")
    fig.savefig(fig_hw03_dir / "pressure_overview.svg", bbox_inches="tight")
    plt.close(fig)
    # Сравнение Boxplot и Violin plot
    fig, (ax_box, ax_viol) = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    ax_box.boxplot(
        [boundary_pressure_mpa, well_pressure_mpa],
        tick_labels=["Граница", "Скважина"],
        patch_artist=True,
        boxprops=dict(facecolor="#ffbb78", color="#ff7f0e"),
        medianprops=dict(color="darkblue", linewidth=1.5),
    )
    ax_box.set_ylabel("Давление, МПа")
    ax_box.set_title("Boxplot")
    ax_box.grid(True, axis="y", alpha=0.3)
    ax_viol.violinplot(
        [boundary_pressure_mpa, well_pressure_mpa],
        showmeans=True,
        showmedians=True,
    )
    ax_viol.set_xticks([1, 2])
    ax_viol.set_xticklabels(["Граница", "Скважина"])
    ax_viol.set_ylabel("Давление, МПа")
    ax_viol.set_title("Violin plot")
    ax_viol.grid(True, axis="y", alpha=0.3)
    fig.savefig(fig_hw03_dir / "distributions.png", dpi=200, bbox_inches="tight")
    fig.savefig(fig_hw03_dir / "distributions.svg", bbox_inches="tight")
    plt.close(fig)
    # Сохранение .npz для ДЗ №4
    viz_data_path = root_dir / "homeworks" / "hw03" / "viz_data.npz"
    np.savez(
        viz_data_path,
        time_h=time_h,
        boundary_pressure_mpa=boundary_pressure_mpa,
        well_pressure_mpa=well_pressure_mpa,
        pressure_drop_mpa=pressure_drop_mpa,
        well_id=well_id,
        radius_m=radius_m,
        observation_pressure_mpa=observation_pressure_mpa,
        observation_drop_mpa=observation_drop_mpa,
    )
    loaded_data = np.load(viz_data_path)
    expected_keys = [
        "time_h",
        "boundary_pressure_mpa",
        "well_pressure_mpa",
        "pressure_drop_mpa",
        "well_id",
        "radius_m",
        "observation_pressure_mpa",
        "observation_drop_mpa",
    ]
    for key in expected_keys:
        assert key in loaded_data.files, f"Ключ {key} отсутствует в .npz"
    print("Все графики ДЗ №3 успешно построены, а файл viz_data.npz проверен!")
if __name__ == "__main__":
    main()
