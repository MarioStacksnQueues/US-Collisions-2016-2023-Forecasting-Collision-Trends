# US Collisions (2016-2023): Forecasting Collision Trends

A comprehensive data science project analyzing and forecasting US traffic collision trends using advanced feature engineering and machine learning models.

**Author:** Mario Cuevas

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Key Features](#key-features)
- [Models](#models)
- [Results](#results)
- [License](#license)

## Overview

This project analyzes 7.7+ million traffic accident records from 2016-2023 to forecast collision trends across the United States. Using advanced feature engineering, time-series analysis, and machine learning techniques, we predict daily accident counts and severity levels.

### Objectives

1. **Exploratory Data Analysis (EDA)** - Understand patterns in US traffic collisions
2. **Feature Engineering** - Create advanced temporal and weather-based features
3. **Baseline Modeling** - Build and compare forecasting models (Prophet, Random Forest)
4. **Severity Classification** - Predict accident severity using environmental factors

## Project Structure

```
US-Collisions-2016-2023-Forecasting-Collision-Trends/
├── README.md                          # Main project description
├── .gitignore                         # Files to ignore
├── requirements.txt                   # Python dependencies
│
├── notebooks/                         # All Jupyter notebooks
│   ├── Week1_EDA.ipynb               # Week 1 exploratory analysis
│   └── Week2_Feature_Engineering.ipynb  # Week 2 modeling
│
├── data/                              # Data files
│   ├── processed/
│   │   ├── daily_accidents_features.csv
│   │   └── model_comparison.csv
│   └── README.md                      # Data description
│
├── reports/                           # Project reports (PDFs)
│   ├── Week1_Report.pdf
│   └── Week2_Report.pdf
│
├── visualizations/                    # HTML/PNG visualizations
│   ├── EDA_Report.html
│   └── US_Accidents_Map.html
│
└── src/                               # Python utility scripts
    └── utils.py                       # Helper functions
```

## Dataset

**Source:** [US Accidents (2016-2023)](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)

**License:** CC-BY-NC-SA-4.0

**Size:** 7.7M+ accident records, 2.9GB uncompressed

**Date Range:** February 2016 - March 2023

**Features Used:**
- Temporal: Date, time, location
- Weather: Temperature, humidity, visibility, wind speed, precipitation
- Severity: Accident severity levels (1-4)

### Citation

```
Moosavi, Sobhan, et al. "A Countrywide Traffic Accident Dataset." (2019).
Moosavi, Sobhan, et al. "Accident Risk Prediction based on Heterogeneous Sparse Data:
New Dataset and Insights." ACM SIGSPATIAL (2019).
```

## Installation

### Prerequisites

- Python 3.9+
- pip package manager
- Kaggle account (for dataset download)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/MarioStacksnQueues/US-Collisions-2016-2023-Forecasting-Collision-Trends.git
cd US-Collisions-2016-2023-Forecasting-Collision-Trends
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Kaggle credentials:
```bash
# Download kaggle.json from https://www.kaggle.com/settings
mkdir -p ~/.kaggle
cp /path/to/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

4. Launch Jupyter:
```bash
jupyter notebook
```

## Usage

### Week 1: Exploratory Data Analysis

Open `notebooks/Week1_EDA.ipynb` to:
- Download and explore the US Accidents dataset
- Analyze accident distribution by state, month, and severity
- Generate interactive visualizations and reports

**Outputs:**
- `visualizations/EDA_Report.html` - SweetViz EDA report
- `visualizations/US_Accidents_Map.html` - Interactive accident map

### Week 2: Feature Engineering & Modeling

Open `notebooks/Week2_Feature_Engineering.ipynb` to:
- Create advanced temporal features (rolling averages, lag variables)
- Generate weather risk index
- Train and compare forecasting models
- Evaluate model performance

**Outputs:**
- `data/processed/daily_accidents_features.csv` - Engineered features dataset
- `data/processed/model_comparison.csv` - Model performance metrics

### Using Utility Functions

```python
from src.utils import (
    load_daily_features,
    create_rolling_features,
    create_lag_features,
    calculate_metrics
)

# Load processed data
df = load_daily_features('data/processed/daily_accidents_features.csv')

# Create custom features
df = create_rolling_features(df, column='Accident_Count', windows=[7, 30])
df = create_lag_features(df, column='Accident_Count', lags=[1, 7, 14])
```

## Key Features

### Engineered Features

1. **Temporal Features**
   - Rolling means (7-day, 30-day)
   - Lag features (1, 3, 7, 14, 30 days)
   - Hour of day, day of week, month, season
   - Weekend and rush hour indicators

2. **Weather Risk Index**
   - Composite score from 5 weather variables
   - Weighted by impact on accident likelihood
   - Formula: `0.3×visibility + 0.3×precip + 0.2×wind + 0.2×humidity`

3. **Statistical Aggregations**
   - Daily accident counts
   - Average severity per day
   - Weather condition averages

## Models

### 1. Facebook Prophet
- **Type:** Time-series forecasting
- **Features:** Trend, seasonality, weather regressors
- **Use Case:** Capturing long-term trends and seasonal patterns

### 2. Random Forest Regressor
- **Type:** Ensemble machine learning
- **Features:** 17 engineered features (weather, lags, rolling stats)
- **Use Case:** Capturing complex non-linear relationships

### 3. Random Forest Classifier (Bonus)
- **Type:** Multi-class classification
- **Target:** Accident severity (1-4)
- **Use Case:** Predicting severity based on conditions

## Results

### Model Comparison

| Model | MAE | RMSE | MAPE (%) | R² |
|-------|-----|------|----------|-----|
| Prophet | ~X.XX | ~X.XX | ~X.XX | ~0.XXX |
| Random Forest | ~X.XX | ~X.XX | ~X.XX | ~0.XXX |

*See `data/processed/model_comparison.csv` for detailed metrics*

### Key Insights

1. **Temporal Patterns:** Accident counts show strong seasonal and weekly patterns
2. **Weather Impact:** Low visibility and high precipitation significantly increase accidents
3. **Lag Importance:** Previous day's accident count is highly predictive
4. **Top States:** CA, TX, FL account for largest share of accidents

## Future Work

- [ ] Implement LSTM/GRU neural networks for sequence modeling
- [ ] Add spatial features (road type, urban/rural classification)
- [ ] Incorporate real-time traffic data
- [ ] Deploy forecasting API
- [ ] State-level granular forecasts

## License

This project uses data under the CC-BY-NC-SA-4.0 license from Kaggle.

## Acknowledgments

- **Dataset:** Sobhan Moosavi et al. (Kaggle)
- **Tools:** Python, Dask, scikit-learn, Prophet, Folium, SweetViz

---

**Contact:** [Your Contact Information]

**Repository:** https://github.com/MarioStacksnQueues/US-Collisions-2016-2023-Forecasting-Collision-Trends
