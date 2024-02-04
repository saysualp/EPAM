import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from omegaconf import OmegaConf
from src.data.make_dataset import make_dataset
from src.features.build_features import build_features
from src.models.train_model import train_model
from src.models.predict_model import predict_model
from src.visualization.visualize import visualize
from pathlib import Path

def load_hydra_config(config_path):
    cfg = OmegaConf.load(config_path)
    return cfg

def load_eda_data():
    comp_dir = Path('data/processed/notebooks')
    return pd.read_csv(comp_dir / 'train.csv', parse_dates=['date'], infer_datetime_format=True)

def run_forecast_flow(cfg, fh_selection, id_selection, show_actuals):
    with st.spinner("Model is training and forecasts are being made, please wait..."):
        make_dataset(cfg)
        model_version = build_features(cfg, forecast_horizon=fh_selection)
        train_model(cfg, model_version=model_version, forecast_horizon=fh_selection)
        predict_model(cfg, model_version=model_version, forecast_horizon=fh_selection)
        fig = visualize(cfg, id=id_selection, model_version=model_version, show_actuals=show_actuals)
        st.markdown("<h3 style='text-align: center;'>Forecast Results</h3>", unsafe_allow_html=True)
        st.plotly_chart(fig)

def render_forecast_tab(cfg):
    fh_options = [28, 42, 60, 90]
    id_options = list(range(1, 541))

    # Create a container for the parameters
    params_container = st.container(border=True)

    # Add the forecast length and family-store pair ID selection widgets to the container
    with params_container:
        col1, col2 = st.columns(2)

        with col1:
            fh_selection = st.select_slider("Select forecast length in days:", options=fh_options, value=fh_options[0])

        with col2:
            id_selection = st.selectbox("Select family-store pair ID:", id_options)

        # Toggle for showing actual values
        show_actuals = st.checkbox("Show actual values", value=False)

        # Display success message if the toggle is activated
        if show_actuals:
            st.success("Actual sales for the forecast period will be shown!", icon="📈")

    if st.button("Forecast"):
        if not id_selection:
            st.error("Please enter a valid ID.")
        else:
            run_forecast_flow(cfg, fh_selection, id_selection, show_actuals)

def render_eda_tab():
    data = load_eda_data()
    
    daily = data.groupby('date').agg({'sales': 'sum', 'dcoilwtico': 'mean'})
    daily['year'] = daily.index.year
    daily['day_of_year'] = daily.index.dayofyear
    daily['smooth7_sales'] = daily['sales'].rolling(window=7,  center=True, min_periods=3).mean()
    daily['smooth30_sales'] = daily['sales'].rolling(window=30,  center=True, min_periods=15).mean()
    daily['smooth365_sales'] = daily['sales'].rolling(window=365,  center=True, min_periods=183).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily.index, y=daily['sales'], mode='lines', name='Daily Sales'))
    fig.add_trace(go.Scatter(x=daily.index, y=daily['smooth7_sales'], mode='lines', name='7-day Moving Average'))
    fig.add_trace(go.Scatter(x=daily.index, y=daily['smooth30_sales'], mode='lines', name='30-day Moving Average'))
    fig.add_trace(go.Scatter(x=daily.index, y=daily['smooth365_sales'], mode='lines', name='365-day Moving Average', line=dict(color='red')))
    fig.update_layout(title='Overall Sales Trend', xaxis_title='Date', yaxis_title='Sales (Qty)')
    st.plotly_chart(fig)

    yoy = daily.groupby(['year', 'day_of_year']).agg({'smooth7_sales': 'sum'}).reset_index()
    fig = px.line(yoy, x='day_of_year', y='smooth7_sales', color='year')
    fig.update_layout(title='YoY Smoothed Sales Plot', xaxis_title='Day of Year', yaxis_title='Sales (Qty)')
    st.plotly_chart(fig)

    pg = data.groupby(['date', 'family']).agg({'sales': 'sum'}).reset_index()
    pg['smooth7_sales'] = pg.groupby('family')['sales'].rolling(window=7, center=True, min_periods=3).mean().reset_index(level=0, drop=True)
    fig = px.line(pg, x='date', y='smooth7_sales', color='family')
    fig.update_layout(title='Smoothed Sales Over Time by Family', xaxis_title='Date', yaxis_title='Sales (Qty)')
    st.plotly_chart(fig)

def main():
    cfg = load_hydra_config('config/config.yaml')

    # Main App Logic
    st.set_page_config(page_title="Sales Prediction Tool")

    tabs = ["Forecast Tool", "Descriptives"]

    st.sidebar.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Using the custom CSS class
    st.sidebar.markdown('<p class="big-font">Navigate</p>', unsafe_allow_html=True)
    tab = st.sidebar.radio("", tabs)

    if tab == "Forecast Tool":
        st.markdown("<h1 style='text-align: center;'>Chain Level Forecast</h1>", unsafe_allow_html=True)
        st.write("The section provides an interactive platform for generating sales forecasts across various family-store pairings within the retail chain.", unsafe_allow_html=True)

        st.info("""
        **Here’s how one can leverage this tool:**

        - **Forecast Length Selection:** Select the desired forecast length in days from slider.

        - **Family-Store Pair Identification:** Each family-store pair within the chain is uniquely identified by an ID number. Through a dropdown menu, select the ID corresponding to the specific family-store pair.

        - **Actual Values Comparison:** An option is available to display actual sales figures alongside the forecasted data for comparison.

        Upon making these selections, clicking on the "Forecast" button generates the sales forecasts.
        """, icon="🔍")
        render_forecast_tab(cfg)

    elif tab == "Descriptives":
        st.markdown("<h1 style='text-align: center;'>Exploratory Data Analysis</h1>", unsafe_allow_html=True)
        st.write("""
        This section highlights some efforts from the exploratory data analysis that give an overall picture of the business's performance.
                
        The aim is to provide a general overview, showing where the business stands and how different segments compare. This includes looking at sales over time, understanding which families are popular, and spotting any significant changes from one period to the next.
        """)
        st.info("""It's important to note that the dashboards presented are just a part of the exploratory analysis. They've been selected to offer useful insights without overwhelming users with too much information.""", icon="📊")
        render_eda_tab()

if __name__ == "__main__":
    main()