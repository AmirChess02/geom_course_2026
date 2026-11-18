from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
def plot_pressure_over_time(table: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        table["time_h"],
        table["boundary_pressure_mpa"],
        label="Давление на границе",
        color="#1f77b4",
        marker="o",
        markersize=4,
        linewidth=1.5,
    )
    ax.plot(
        table["time_h"],
        table["well_pressure_mpa"],
        label="Давление в скважине",
        color="#ff7f0e",
        marker="s",
        markersize=4,
        linewidth=1.5,
    )
    ax.set_xlabel("Время, ч")
    ax.set_ylabel("Давление, МПа")
    ax.set_title("Изменение давления во времени")
    ax.legend()
    ax.grid(True, alpha=0.3)
    min_index = table["well_pressure_mpa"].idxmin()
    min_time = table.loc[min_index, "time_h"]
    min_pressure = table.loc[min_index, "well_pressure_mpa"]
    ax.annotate(
        f"Мин: {min_pressure:.2f} МПа",
        xy=(min_time, min_pressure),
        xytext=(min_time + 5, min_pressure - 0.5),
        arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=6),
        fontsize=9,
    )
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "pressure_over_time.png", dpi=200, bbox_inches="tight")
    fig.savefig(output_dir / "pressure_over_time.svg", bbox_inches="tight")
    plt.close(fig)
def plot_pressure_by_radius(table: pd.DataFrame, output_dir: Path) -> None:
    assert (table["radius_m"] > 0).all(), "Значения радиуса должны быть больше нуля!"
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        table["radius_m"],
        table["pressure_mpa"],
        linestyle="none",
        marker="o",
        color="#2ca02c",
        markersize=6,
        label="Наблюдательные скважины",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Расстояние, м")
    ax.set_ylabel("Давление, МПа")
    ax.set_title("Давление в наблюдательных скважинах")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "pressure_by_radius.png", dpi=200, bbox_inches="tight")
    fig.savefig(output_dir / "pressure_by_radius.svg", bbox_inches="tight")
    plt.close(fig)
