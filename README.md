Chain Level Forecast
==============================

Executive Summary
------------

#### Problem Statement
This project focuses on predicting sales for numerous product families across Favorita stores located in Ecuador. By utilizing training data that encompasses dates, details on stores and products, promotional statuses, and actual sales figures, the goal is to accurately forecast future sales. The forecast takes into account various influencing factors, including promotions and external economic factors.

#### Data
- **train.csv**: Contains time series data of features such as `store_nbr`, `family`, and `onpromotion`, along with the target variable `sales`.
  - `store_nbr` denotes the store where products are sold.
  - `family` indicates the product type.
  - `sales` represent total sales of a product family at a specific store on a certain date, with fractional values possible.
  - `onpromotion` shows the number of items in a product family that were on promotion at a store on a specific date.

- **stores.csv**: Provides metadata about stores, including `city`, `state`, `type`, and `cluster`. The `cluster` attribute groups similar stores together.

- **oil.csv**: Lists daily oil prices, an important economic indicator for Ecuador, covering both the training and testing periods.

**Additional Notes**: Factors such as bi-weekly public sector wage payments and external events like the 2016 Ecuador earthquake are considered for their potential impact on sales.

#### Approach
The project adopts a comprehensive approach involving:
1. **Exploratory Data Analysis (EDA)** to understand the data.
2. **Feature Engineering** to prepare the data for modeling.
3. **Model Training** focusing on tranining individual family-store combinations using LightGBM, employing a recursive strategy.

#### Results
The model's performance is evaluated using MAPE (Mean Absolute Percentage Error) and RMSE (Root Mean Square Error), achieving around 23% MAPE and 215 RMSE. These metrics indicate the model's consistency and robustness through backtesting.

#### Tech Stack
- **Repo Building**: Utilizes Cookiecutter for structured project setup.
- **Config Management**: Employs Hydra for flexible configuration management.
- **Code Quality**: Maintains code quality through Flake8 and Black.
- **Application Hosting**: Presents the application via Streamlit for interactive user engagement.


Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── config             <- Contains configs consumed
    │   ├── config.yaml    <- Shelf config for forecast pipeline 
    │
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │ 
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks
    │
    ├── requirements.in   <- The requirements file
    │                         
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment,
    │                         generated with `pip-compile requirements.in`
    │
    ├── setup.py           <- Makes project pip installable (pip install -e .) so src can be imported
    │
    ├── scripts            <- Runnable scripts
    │   ├── run.py         <- Using the config file orchestrates forecast pipeline 
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    ├── tests              <- Contains some unit tests
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

Usage
------------
This project provides two main methods for users to access sales forecast results:

#### 1. Generating Forecasts Locally
Users interested in running the forecast pipeline locally to generate predictions can do so by executing the following command in the terminal:

`python3 -m scripts.run`

This command triggers the forecasting script, which then processes the data according to the configurations set in `config.yaml`. As a result, it outputs the predicted sales for each product family across all Favorita stores.

#### 2. Interactive Web Application
For those seeking an interactive experience, an online application has been developed using Streamlit. This application allows users to view and interact with the sales forecasts directly through a web browser.

The web application can be accessed here: [Chain Level Forecast App](https://chainlevelforecast.streamlit.app)

The application features a user-friendly interface, enabling users to select specific stores and product families to visualize their sales forecasts. It is designed to be intuitive, allowing for easy navigation through the data and providing insights into future sales trends without the necessity of running any local code.

References
------------
Corporación Favorita, inversion, Julia Elliott, Mark McDonald. (2017). Corporación Favorita Grocery Sales Forecasting. [View on Kaggle](https://kaggle.com/competitions/favorita-grocery-sales-forecasting)
