# US Traffic Accident Forecasting 🚗📊

Deep learning models for predicting daily traffic accident counts using 2016-2023 US accident data.

![Model Improvements](results/week3_improvements.png)

## 🎯 Key Results

| Model | R² Score | Improvement | Status |
|-------|----------|-------------|--------|
| **GRU (Optimized)** | **0.415** | **+128%** | ✅ Best DL Model |
| LSTM + Attention | 0.297 | +270% | ✅ Strong |
| TCN | 0.201 | +264% | ✅ Good |
| Transformer | 0.027 | +244% | ⚠️ Needs Work |
| Random Forest (Baseline) | 0.55 | - | 🎯 Target |

**Achievement:** Improved GRU from R² = 0.182 → 0.415 in 2 hours through configuration optimization!

## 📊 Dataset

### Source
- **Dataset:** [US Accidents (2016-2023) - Kaggle](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)
- **Original Size:** 7.7M accident records, 49 states (2.9GB)
- **License:** CC-BY-NC-SA-4.0

### Processed Data
- **File:** `data/processed/daily_accidents_features.csv`
- **Size:** 2,568 daily observations
- **Date Range:** 2016-01-14 to 2023-03-31
- **Features:** 23 engineered features including:
  - Temporal: Daily counts, moving averages (7d, 30d), lag features (1, 3, 7, 14, 30 days)
  - Weather: Temperature, humidity, visibility, wind speed, precipitation
  - Derived: Weather Risk Index, standard deviations

### Citation
Moosavi, Sobhan, et al. "A Countrywide Traffic Accident Dataset.", 2019.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Google Colab (recommended) or Jupyter

### Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/us-traffic-accident-forecasting.git
cd us-traffic-accident-forecasting

# Install dependencies
pip install -r requirements.txt
```

### Download Data

1. Download from [Kaggle](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)
2. Or use the processed data in `data/processed/`

### Run Notebooks

```bash
# Start Jupyter
jupyter notebook

# Open notebooks in this order:
# 1. notebooks/Week1_EDA.ipynb
# 2. notebooks/Week2_Feature_Engineering.ipynb
# 3. notebooks/Week3_DeepLearning.ipynb
```

## 🔬 Methodology

### Week 1: Exploratory Data Analysis
- Analyzed 7.7M accident records
- Identified temporal patterns (winter peaks, COVID impact)
- Geographic distribution across 49 states
- Created interactive visualizations

### Week 2: Feature Engineering & Baselines
- **Random Forest:** R² = 0.55, MAE = 662 ✓
- **Prophet:** R² = 0.14 (struggled with COVID disruption)
- Engineered 23 features (lags, rolling averages, weather index)
- Key finding: Lag_1 (yesterday's count) most predictive (r = 0.82)

### Week 3: Deep Learning Models

**Initial Implementation (7-day forecast, 30-day lookback):**
- GRU: R² = 0.182
- Other models: Negative R² scores

**Optimized Configuration (1-day forecast, 14-day lookback):**

| Model | Architecture | R² | MAE |
|-------|-------------|-----|-----|
| **GRU** | Bidirectional GRU (64 units) | **0.415** | 0.560 |
| LSTM + Attention | BiLSTM (128→64) + Attention | 0.297 | 0.636 |
| TCN | 4 dilated conv blocks | 0.201 | 0.650 |
| Transformer | 2 encoder blocks, 4 heads | 0.027 | 0.808 |

**Quick Fixes Applied:**
1. Forecast horizon: 7 days → 1 day (much easier task)
2. Lookback window: 30 days → 14 days (reduced overfitting)
3. Training epochs: 50 → 100 (better convergence)
4. Simplified GRU architecture (85K params vs 200K)

**Results:** GRU reached 75% of RF baseline!

### Week 4: Hyperparameter Optimization (Planned)
- Optuna automated search
- Target: R² > 0.60 (beat baseline!)
- Feature ablation studies

## 🛠️ Tech Stack

- **Python 3.8+**
- **Deep Learning:** TensorFlow/Keras
- **ML:** Scikit-learn, Facebook Prophet
- **Data:** Pandas, NumPy, Dask
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Optimization:** Optuna

## 📁 Project Structure

```
us-traffic-accident-forecasting/
│
├── data/
│   └── processed/
│       └── daily_accidents_features.csv    # Engineered features
│
├── notebooks/
│   ├── Week1_EDA.ipynb                     # Exploratory analysis
│   ├── Week2_Feature_Engineering.ipynb     # Baseline models
│   └── Week3_DeepLearning.ipynb            # Deep learning models
│
├── models/
│   ├── gru_model.h5                        # Best GRU model
│   └── tcn_model.h5                        # TCN model
│
├── results/
│   ├── model_comparison_results.csv        # Performance metrics
│   ├── model_predictions.csv               # Sample predictions
│   └── week3_improvements.png              # Before/after visualization
│
├── src/
│   └── utils.py                            # Helper functions
│
├── visualizations/
│   ├── EDA_Report.html                     # Interactive EDA
│   └── US_Accidents_Map.html               # Geographic visualization
│
├── .gitignore
├── README.md                               # This file
└── requirements.txt                        # Dependencies
```

## 📈 Results Summary

### Performance Progress

```
Week 2 Baseline (Random Forest):
  ████████████████████████████ R² = 0.55

Week 3 Initial (GRU, no optimization):
  ████░░░░░░░░░░░░░░░░░░░░░░░░ R² = 0.18

Week 3 Optimized (GRU, quick fixes):
  ███████████████████░░░░░░░░░ R² = 0.42  (75% of baseline!)
```

### Key Findings

✅ **1-day forecasting** >> multi-day (R² 0.415 vs 0.182)  
✅ **Lag features** critical (Lag_1 correlation = 0.82)  
✅ **Model complexity** must match data size  
✅ **Deep learning viable** when properly configured  

## 🎯 Future Work

### Week 4 (In Progress)
- [ ] Hyperparameter optimization using Optuna
- [ ] Target: R² > 0.60 (beat Random Forest baseline)
- [ ] Feature ablation studies
- [ ] Final technical report

### Long-term
- [ ] Multi-step forecasting (3, 7, 14 days)
- [ ] State-level models
- [ ] Real-time prediction dashboard
- [ ] Weather API integration

## 🔍 Key Insights

1. **Forecast Horizon Matters Most**
   - Changing 7-day → 1-day: +0.20 R² gain
   - Single biggest improvement factor

2. **Yesterday Predicts Tomorrow**
   - Lag_1 feature: r = 0.82 correlation
   - Traffic patterns are highly autocorrelated

3. **Simpler Can Be Better**
   - Reduced GRU from 200K → 85K parameters
   - Improved R² from 0.18 → 0.42
   - Lesson: Match model to data size

4. **Deep Learning Requires Tuning**
   - Default configs often fail
   - Configuration >> Architecture choice
   - Week 4 optimization will push past baseline

## 📊 Visualizations

Interactive visualizations available in `visualizations/`:

- **EDA_Report.html:** Comprehensive exploratory analysis
- **US_Accidents_Map.html:** Geographic distribution
- **week3_improvements.png:** Model performance comparison

## 🤝 Contributing

This is an academic project. Feedback and suggestions welcome!

## 📄 License

MIT License - Free to use for educational purposes

## 👤 Author

**Mario Cuevas**
- Course: Machine Learning Project
- Date: November 2024
- Focus: Time Series Forecasting with Deep Learning

## 🙏 Acknowledgments

- Dataset: Moosavi et al. (Kaggle)
- Course instructors and peers
- TensorFlow/Keras community

---

⭐ **Star this repo if helpful!**

📊 **Status:** Week 3 Complete ✅ | Week 4 In Progress 🚧

🎯 **Goal:** Beat R² = 0.55 baseline with optimized deep learning
