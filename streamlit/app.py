# importing all required libraries
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import plotly.express as px

# loading cleaned data
df=pd.read_csv('data/cleaned/charging_clean.csv', parse_dates=['StartDatetime','EndDatetime'])
df['Hour']=df['StartDatetime'].dt.hour
df['Weekday']=df['StartDatetime'].dt.day_name()
df['Month']=df['StartDatetime'].dt.to_period('M').astype(str)

# loading saved models from 09 notebook
reg_model=joblib.load('models/regression_model.pkl')
clf_model=joblib.load('models/classification_model.pkl')
ts_model=joblib.load('models/timeseries_model.pkl')
threshold=joblib.load('models/high_demand_threshold.pkl')

# Sidebar navigation
st.sidebar.title('EV Charging Insights')
option=st.sidebar.selectbox('select a section:',
                            ['Home','Overview','Demand Patterns','Prediction','Forecast','Model Insights'])

# Sections
if option=='Home':
    st.title('EV Charging Demand Prediction and Usage Pattern Analysis')
    st.markdown("""
    A dashboard analysing **EV charging sessions** (~72,000 sessions, Oct 2021 - Sep 2022).

    Objective is to understand charging patterns, predict the energy demand of a session,
    flag high-demand sessions and forecast daily charging demand.""")

elif option=='Overview':
    st.title('Overview')

    # First row of KPI cards
    col1,col2,col3=st.columns(3)
    col1.metric('Total Sessions', f"{len(df):,}")
    col2.metric('Total Demand (kWh)', f"{df['Demand'].sum():,.0f}")
    col3.metric('Avg Demand per Session (kWh)', f"{df['Demand'].mean():,.2f}")

    # Second row of KPI cards
    col4,col5,col6=st.columns(3)
    col4.metric('Avg Duration (min)', f"{df['Duration'].mean():,.0f}")
    col5.metric('Total Chargers', f"{df['ChargerID'].nunique():,}")
    col6.metric('Locations', f"{df['Location'].nunique()}")

    # Sessions by location
    st.subheader('Sessions by Location')
    loc_sessions=df['Location'].value_counts().reset_index()
    loc_plot=px.bar(loc_sessions, x='count', y='Location', orientation='h', title='Number of Sessions by Location')
    loc_plot.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(loc_plot)

    # Monthly total demand (first month dropped as it has only one partial day)
    st.subheader('Monthly Total Demand')
    monthly=df.groupby('Month')['Demand'].sum().reset_index().iloc[1:]
    monthly_plot=px.line(monthly, x='Month', y='Demand', markers=True, title='Total Demand per Month (kWh)')
    st.plotly_chart(monthly_plot)
elif option=='Demand Patterns':
    st.title('Demand Patterns')
    tab1,tab2,tab3,tab4=st.tabs(['By Hour','By Weekday','By Location','By Charger Type'])

    with tab1:
        # Number of sessions by start hour
        st.subheader('Sessions by Start Hour')
        hour_sessions=df['Hour'].value_counts().sort_index().reset_index()
        hour_plot=px.bar(hour_sessions, x='Hour', y='count', title='Number of Sessions by Start Hour')
        st.plotly_chart(hour_plot)
        st.caption('Most sessions start in the evening, so this is the busiest time for the chargers.')

    with tab2:
        # Total demand by weekday
        st.subheader('Total Demand by Weekday')
        weekday_demand=df.groupby('Weekday')['Demand'].sum().sort_values(ascending=False).reset_index()
        weekday_plot=px.bar(weekday_demand, x='Weekday', y='Demand', title='Total Demand by Weekday (kWh)')
        st.plotly_chart(weekday_plot)
        st.caption('Weekdays are busier than weekends and Sunday is the lowest day.')

    with tab3:
        # Average demand per session by location
        st.subheader('Average Demand per Session by Location')
        loc_demand=df.groupby('Location')['Demand'].mean().reset_index()
        loc_demand_plot=px.bar(loc_demand, x='Demand', y='Location', orientation='h',
                               title='Average Demand per Session by Location (kWh)')
        loc_demand_plot.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(loc_demand_plot)

    with tab4:
        # Sessions and average demand by charger type
        st.subheader('Charger Type Comparison')
        type_sessions=df['ChargerType'].value_counts().reset_index()
        type_plot=px.pie(type_sessions, names='ChargerType', values='count', title='Share of Sessions by Charger Type')
        st.plotly_chart(type_plot)

        type_demand=df.groupby('ChargerType')['Demand'].mean().reset_index()
        type_demand_plot=px.bar(type_demand, x='ChargerType', y='Demand', title='Average Demand per Session by Charger Type (kWh)')
        st.plotly_chart(type_demand_plot)
elif option=='Prediction':
    st.title('Session Demand Prediction')
    st.markdown('Enter the session details to predict the energy demand and whether it will be a High Demand session.')

    # Lists for the dropdowns
    locations=sorted(df['Location'].unique())
    companies=sorted(df['ChargerCompany'].unique())
    weekdays=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

    # Session details (left) and history details (right)
    col1,col2=st.columns(2)
    location=col1.selectbox('Location', locations)
    charger_type=col1.selectbox('Charger Type', [0,1])
    company=col1.selectbox('Charger Company', companies)
    hour=col1.number_input('Start Hour (0-23)', min_value=0, max_value=23, value=18)
    weekday=col1.selectbox('Weekday', weekdays)

    guest=col2.selectbox('Guest User?', ['No','Yes'])
    user_sessions=col2.number_input('User Past Sessions', min_value=0, value=20)
    user_avg=col2.number_input('User Past Avg Demand (kWh)', min_value=0.0, value=15.0)
    charger_sessions=col2.number_input('Charger Past Sessions', min_value=0, value=50)
    charger_avg=col2.number_input('Charger Past Avg Demand (kWh)', min_value=0.0, value=15.0)

    # Converting inputs to the same format used in model training
    weekday_num=weekdays.index(weekday)
    is_weekend=int(weekday_num>=5)
    is_guest=int(guest=='Yes')

    # Guests have no personal history (same as feature engineering notebook)
    if is_guest==1:
        user_sessions=0
        user_avg=np.nan

    # One row with the same columns the models were trained on
    input_df=pd.DataFrame({'ChargerCompany':[company], 'ChargerType':[charger_type], 'Hour':[hour],
                           'Weekday':[weekday_num], 'IsWeekend':[is_weekend], 'IsGuest':[is_guest],
                           'UserPastSessions':[user_sessions], 'UserPastAvgDemand':[user_avg],
                           'ChargerPastSessions':[charger_sessions], 'ChargerPastAvgDemand':[charger_avg],
                           'Location':[location]})

    if st.button('Predict'):
        pred_demand=reg_model.predict(input_df)[0]
        pred_class=clf_model.predict(input_df)[0]
        prob_high=clf_model.predict_proba(input_df)[0][1]

        col3,col4,col5=st.columns(3)
        col3.metric('Predicted Demand (kWh)', f"{pred_demand:.2f}")
        if pred_class==1:
            col4.metric('Session Type', 'High Demand')
        else:
            col4.metric('Session Type', 'Normal Demand')
        col5.metric('Chance of High Demand', f"{prob_high*100:.0f}%")
        st.caption(f"High Demand means the session uses {threshold} kWh or more.")
elif option=='Forecast':
    st.title('Daily Demand Forecast')
    st.markdown('Forecast of total daily charging demand using Holt-Winters (weekly seasonality), trained on Oct 2021 - Sep 2022.')

    # Number of days to forecast
    days=st.slider('Number of days to forecast', min_value=7, max_value=60, value=30)

    # Forecast table
    forecast=ts_model.forecast(days).reset_index()
    forecast.columns=['Date','Forecast Demand (kWh)']
    forecast['Forecast Demand (kWh)']=forecast['Forecast Demand (kWh)'].round(1)

    # KPI cards
    col1,col2=st.columns(2)
    col1.metric('Total Forecast Demand (kWh)', f"{forecast['Forecast Demand (kWh)'].sum():,.0f}")
    col2.metric('Avg Daily Forecast (kWh)', f"{forecast['Forecast Demand (kWh)'].mean():,.0f}")

    # Forecast chart
    forecast_plot=px.line(forecast, x='Date', y='Forecast Demand (kWh)', markers=True, title='Forecast of Daily Demand')
    st.plotly_chart(forecast_plot)

    # Forecast values
    st.subheader('Forecast Values')
    st.dataframe(forecast)
    st.caption('The weekly pattern repeats in the forecast: lower demand on weekends, higher on weekdays.')
elif option=='Model Insights':
    st.title('Model Insights')
    tab1,tab2,tab3=st.tabs(['Regression','Classification','Time Series'])

    with tab1:
        # Final regression model test results (from 06 notebook)
        st.subheader('Regression - LightGBM (Test Results)')
        reg_results=pd.DataFrame({'Metric':['MAE','RMSE','R2'], 'Value':[8.12, 11.16, 0.354]})
        st.dataframe(reg_results)

        # Feature importance from the saved regression model
        reg_features=reg_model.named_steps['preprocess'].get_feature_names_out()
        reg_gain=reg_model.named_steps['model'].booster_.feature_importance(importance_type='gain')
        reg_importance=pd.DataFrame({'Feature':reg_features, 'Importance %':reg_gain/reg_gain.sum()*100})
        reg_importance=reg_importance.sort_values('Importance %', ascending=False).head(10)
        reg_imp_plot=px.bar(reg_importance, x='Importance %', y='Feature', orientation='h', title='Top 10 Features - Regression')
        reg_imp_plot.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(reg_imp_plot)

    with tab2:
        # Final classification model test results (from 07 notebook)
        st.subheader('Classification - LightGBM (Test Results)')
        clf_results=pd.DataFrame({'Metric':['Accuracy','Precision','Recall','F1','ROC-AUC'],
                                  'Value':[0.726, 0.692, 0.789, 0.737, 0.806]})
        st.dataframe(clf_results)

        # Feature importance from the saved classification model
        clf_features=clf_model.named_steps['preprocess'].get_feature_names_out()
        clf_gain=clf_model.named_steps['model'].booster_.feature_importance(importance_type='gain')
        clf_importance=pd.DataFrame({'Feature':clf_features, 'Importance %':clf_gain/clf_gain.sum()*100})
        clf_importance=clf_importance.sort_values('Importance %', ascending=False).head(10)
        clf_imp_plot=px.bar(clf_importance, x='Importance %', y='Feature', orientation='h', title='Top 10 Features - Classification')
        clf_imp_plot.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(clf_imp_plot)

    with tab3:
        # 30-day forecast results on September 2022 (from 08 notebook)
        st.subheader('Time Series Models (30-day Test)')
        ts_results=pd.DataFrame({'Model':['Naive','Moving Average (7)','Holt-Winters (seasonal only)','Simple Exp Smoothing',
                                          'Holt-Winters (trend + seasonal)','Holt (trend)'],
                                 'MAE':[514.5, 580.2, 576.1, 617.4, 696.2, 742.6],
                                 'RMSE':[835.7, 891.9, 866.0, 923.7, 951.9, 1012.9]})
        st.dataframe(ts_results)
        st.caption('Simple models did better because demand dropped in September after summer growth. '
                   'Holt-Winters (seasonal only) was chosen as it is close to the best and keeps the weekly pattern.')