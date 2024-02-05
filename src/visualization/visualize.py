import pickle
from pathlib import Path
from omegaconf import DictConfig
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def visualize(
    cfg: DictConfig, id: int, model_version: str, show_actuals=False
):
    """
    Visualizes the output comparing
    with history (last 365 days, if exists) and future
    """
    # Construct paths to the input and output directories
    input_dir = Path(cfg.paths.interim_data_path) / model_version

    # Make paths
    y_train_path = input_dir / "y_train.pkl"
    y_pred_path = input_dir / "y_preds.pkl"
    y_holdout_path = input_dir / "y_holdout.pkl"

    with open(y_train_path, "rb") as file:
        y_train = pickle.load(file)

    with open(y_pred_path, "rb") as file:
        y_pred = pickle.load(file)

    with open(y_holdout_path, "rb") as file:
        y_holdout = pickle.load(file)

    # History
    history_dates = y_train[id].time_index[-365:]
    history_values = y_train[id].values()[-365:]

    # Actuals (from y_valid)
    actual_dates = y_holdout[id].time_index
    actual_values = y_holdout[id].values()

    # Forecasts (from y_pred)
    forecast_dates = y_pred[id].time_index
    forecast_values = y_pred[id].values()

    # Initialize figure with subplots
    fig = make_subplots(specs=[[{"secondary_y": False}]])
    fig.add_trace(
        go.Scatter(
            x=history_dates, y=history_values.flatten(), name="History"
        ),
        secondary_y=False,
    )
    if show_actuals:
        fig.add_trace(
            go.Scatter(
                x=actual_dates, y=actual_values.flatten(), name="Actual"
            ),
            secondary_y=False,
        )
    fig.add_trace(
        go.Scatter(
            x=forecast_dates, y=forecast_values.flatten(), name="Forecast"
        ),
        secondary_y=False,
    )
    fig.update_layout(title_text="Results for selected Family-Store pair")
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Sales (Qty)", secondary_y=False)

    return fig
