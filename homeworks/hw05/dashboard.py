from pathlib import Path
from dash import Dash, Input, Output, dcc, html
import pandas as pd
import plotly.express as px
root_dir = Path(__file__).resolve().parents[2]
csv_path = root_dir / "homeworks" / "hw05" / "pressure_long.csv"
df = pd.read_csv(csv_path)
time_values = sorted(df["time_h"].unique())
marks_dict = {i: f"{t:.1f}h" for i, t in enumerate(time_values)}
app = Dash(__name__)
app.layout = html.Div(
    [
        html.H1("Интерактивный просмотр давления по скважинам", style={"textAlign": "center"}),
        html.Div(
            [
                html.Label("Выберите момент времени:"),
                dcc.Slider(
                    id="time-slider",
                    min=0,
                    max=len(time_values) - 1,
                    step=1,
                    value=0,
                    marks=marks_dict,
                ),
            ],
            style={"width": "80%", "margin": "0 auto", "padding": "20px"},
        ),
        html.Div(
            [
                dcc.Graph(id="pressure-graph"),
            ],
            style={"width": "80%", "margin": "0 auto"},
        ),
    ]
)
@app.callback(
    Output("pressure-graph", "figure"),
    Input("time-slider", "value"),
)
def update_graph(slider_index):
    selected_time = time_values[slider_index]
    dff = df[df["time_h"] == selected_time]
    fig = px.line(
        dff,
        x="radius_m",
        y="pressure_mpa",
        markers=True,
        title=f"Распределение давления при t = {selected_time:.1f} ч",
        labels={"radius_m": "Расстояние, м", "pressure_mpa": "Давление, МПа"},
    )
    fig.update_xaxes(type="log")
    fig.update_layout(transition_duration=300)
    return fig
if __name__ == "__main__":
    app.run(debug=True, port=8050)
