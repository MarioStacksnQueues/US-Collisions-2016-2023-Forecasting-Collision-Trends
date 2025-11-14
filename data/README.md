# Data Directory

This directory contains all data files used in the US Collisions Forecasting project.

## Directory Structure

```
data/
├── processed/          # Processed and engineered datasets
│   ├── daily_accidents_features.csv
│   └── model_comparison.csv
└── README.md          # This file
```

## Processed Data Files

### daily_accidents_features.csv

**Description:** Daily aggregated accident statistics with engineered features for time-series forecasting.

**Shape:** 2,568 rows × 23 columns

**Date Range:** 2016-01-14 to 2023-03-31

**Features:**
1. `Date` - Date of observation
2. `Accident_Count` - Total number of accidents per day
3. `Avg_Severity` - Average severity level (1-4)
4. `Avg_Temperature` - Average temperature (°F)
5. `Avg_Humidity` - Average humidity (%)
6. `Avg_Visibility` - Average visibility (miles)
7. `Avg_Wind_Speed` - Average wind speed (mph)
8. `Avg_Precipitation` - Average precipitation (inches)
9. `Accident_Count_7d_MA` - 7-day moving average of accident count
10. `Accident_Count_30d_MA` - 30-day moving average of accident count
11. `Accident_Count_7d_Std` - 7-day standard deviation of accident count
12. `Accident_Count_Lag_1` - Accident count lagged by 1 day
13. `Accident_Count_Lag_3` - Accident count lagged by 3 days
14. `Accident_Count_Lag_7` - Accident count lagged by 7 days
15. `Accident_Count_Lag_14` - Accident count lagged by 14 days
16. `Accident_Count_Lag_30` - Accident count lagged by 30 days
17. `Avg_Temperature_Lag_1` - Temperature lagged by 1 day
18. `Avg_Humidity_Lag_1` - Humidity lagged by 1 day
19. `Avg_Visibility_Lag_1` - Visibility lagged by 1 day
20. `Avg_Temperature_Lag_7` - Temperature lagged by 7 days
21. `Avg_Humidity_Lag_7` - Humidity lagged by 7 days
22. `Avg_Visibility_Lag_7` - Visibility lagged by 7 days
23. `Weather_Risk_Index` - Composite weather risk score

**Usage:**
- Used for time-series forecasting models (Prophet, Random Forest)
- Contains engineered features for predictive modeling

---

### model_comparison.csv

**Description:** Performance metrics comparison for baseline forecasting models.

**Models Evaluated:**
- Facebook Prophet
- Random Forest Regressor

**Metrics:**
- MAE (Mean Absolute Error)
- RMSE (Root Mean Square Error)
- MAPE (Mean Absolute Percentage Error)
- R² (R-squared/Coefficient of Determination)

**Usage:**
- Model selection and performance comparison
- Baseline benchmark for future models

---

## Data Source

**Original Dataset:** [US Accidents (2016-2023)](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)

**License:** CC-BY-NC-SA-4.0

**Citation:**
- Moosavi, Sobhan, Mohammad Hossein Samavatian, Srinivasan Parthasarathy, and Rajiv Ramnath. "A Countrywide Traffic Accident Dataset.", 2019.
- Moosavi, Sobhan, Mohammad Hossein Samavatian, Srinivasan Parthasarathy, Radu Teodorescu, and Rajiv Ramnath. "Accident Risk Prediction based on Heterogeneous Sparse Data: New Dataset and Insights." In proceedings of the 27th ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems, ACM, 2019.

**Original Size:** ~7.7M accident records (2.9GB uncompressed)

**Note:** The `data/processed/` folder contains only the aggregated and feature-engineered datasets. The raw dataset should be downloaded separately from Kaggle if needed.
