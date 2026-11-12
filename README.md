# EV Charging Demand Prediction and Usage Pattern Analysis

Data analysis and machine learning on EV charging sessions (about 72,000 sessions, Sep 2021 – Sep 2022) using Python, Pandas, scikit-learn, LightGBM, statsmodels and Streamlit.

## Objective
Find charging usage patterns, predict the energy demand of a session, classify high-demand sessions and forecast daily demand.

## Approach
1. Understood the dataset and created a data dictionary
2. Checked data quality and cleaned the data with Pandas
3. Did EDA (univariate, bivariate, trend, multivariate)
4. Created features: hour, weekday, weekend flag, guest flag, user and charger past sessions and past average demand
5. Built regression models to predict Demand (kWh)
6. Built classification models to predict High vs Normal demand
7. Did time series analysis and forecast daily demand
8. Saved the final models and built a Streamlit dashboard

## Methodology
- **Cleaning:** removed duplicates, zero and negative durations and impossible charging speeds; dropped redundant date/time columns
- **Modelling:** time-based train-test split, pipelines, TimeSeriesSplit cross-validation, RandomizedSearchCV tuning, SHAP and LIME; Duration left out to avoid data leakage
- **Time series:** daily resampling, decomposition, ADF test, Moving Average and Exponential Smoothing models
- **Dashboard:** 6 sections, including demand prediction and forecast

## Key Findings
1. Demand is right-skewed: median 14.26 kWh, mean 17.57 kWh.
2. Chargers are busiest at 5–6 pm, and each location has its own peak hours.
3. Charger type 1 is likely the fast charger (about 33 kW vs 6.6 kW).
4. Demand roughly doubled over the year, with Sunday the lowest day of the week.
5. A user's past average demand is the strongest predictor of demand.
6. LightGBM was the best model: regression test MAE 8.12 kWh, classification ROC-AUC 0.806.

## How to Run
1. `pip install -r requirements.txt`
2. Put `ChargingRecords.csv` in `data/raw/` (create `data/raw/` and `data/cleaned/`)
3. Run notebooks 01 to 09 in order
4. `streamlit run streamlit/app.py`
