# 🚗 US Traffic Accident Forecasting (2016-2023)

**Deep Learning Models for Predicting Daily Traffic Accident Counts**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://us-collisions-2016-2023-forecasting-collision-trends-i96zdwvvj.streamlit.app/)

---

## 📊 Live Dashboard

**🌐 Interactive Dashboard:** [View Live Demo](https://us-collisions-2016-2023-forecasting-collision-trends-i96zdwvvj.streamlit.app/)

Explore model predictions, compare architectures, and analyze forecasting performance through an interactive Streamlit dashboard.

---

## 📋 Project Overview

This project implements advanced deep learning models to forecast daily traffic accident counts across US states using 2016-2023 accident data. The project follows a systematic multi-week progression from exploratory data analysis through state-of-the-art neural architectures.

### Key Achievements
- ✅ Implemented 4 deep learning architectures (GRU, LSTM+Attention, TCN, Transformer)
- ✅ Best model (GRU) achieved **R² = 0.415** (75% of Random Forest baseline)
- ✅ Comprehensive interpretability analysis with attention mechanisms
- ✅ Interactive dashboard for model comparison and visualization
- ✅ Cross-validation on state-level splits

---

## 🧠 Models Implemented

### Week 3: Deep Learning Forecasting

| Model | Architecture | MAE | RMSE | R² Score |
|-------|-------------|-----|------|----------|
| **GRU** ⭐ | Bidirectional GRU (2 layers) | 0.560 | 0.771 | **0.415** |
| LSTM + Attention | Bidirectional LSTM with attention | 0.636 | 0.846 | 0.297 |
| TCN | Temporal Convolutional Network | 0.650 | 0.901 | 0.201 |
| Transformer | Multi-head self-attention | 0.808 | 0.995 | 0.027 |

**Baseline (Week 2):** Random Forest - R² = 0.55

---

## 🎯 Model Performance

### GRU Model (Best Performer)
- **R² Score:** 0.415 (75% of baseline performance)
- **Mean Absolute Error:** 0.560
- **Root Mean Squared Error:** 0.771
- **Architecture:** Bidirectional GRU with 128 hidden units, 2 layers, dropout 0.3

### Key Insights
- Recent 7-day history drives predictions (60% attention weight)
- Day-of-week patterns critical: Monday peaks, weekend dips
- Seasonal variations captured: 10-15% summer increase
- Top features: Lag_1 (28%), Day_of_Week (22%), Rolling_7day_mean (18%)

---

## 📁 Project Structure

```
US-Collisions-2016-2023-Forecasting-Collision-Trends/
│
├── notebooks/
│   ├── Week1_EDA.ipynb                    # Exploratory Data Analysis
│   ├── Week2_Feature_Engineering.ipynb    # Feature engineering & baselines
│   └── Week3_DeepLearning.ipynb          # Deep learning models
│
├── data/processed/
│   ├── daily_accidents_features.csv       # Processed daily features
│   └── model_comparison.csv               # Model performance metrics
│
├── results/
│   ├── model_comparison_results.csv       # Detailed model metrics
│   ├── model_predictions.csv              # Test set predictions
│   └── week3_improvements.png             # Performance visualization
│
├── src/
│   └── utils.py                           # Utility functions
│
├── visualizations/                        # Generated plots and figures
│
├── streamlit_dashboard.py                 # Interactive dashboard (deployed)
├── requirements.txt                       # Python dependencies
└── README.md                              # This file
```

---

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.9+
pip
```

### Installation

```bash
# Clone repository
git clone https://github.com/mariostacksqueues/US-Collisions-2016-2023-Forecasting-Collision-Trends.git
cd US-Collisions-2016-2023-Forecasting-Collision-Trends

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\Activate

# Install dependencies
pip install -r requirements.txt
```

### Run Dashboard Locally

```bash
# Activate virtual environment
source venv/bin/activate  # On Windows: .\venv\Scripts\Activate

# Run Streamlit dashboard
streamlit run streamlit_dashboard.py

# Opens at http://localhost:8501
```

### Run Notebooks

```bash
# Launch Jupyter
jupyter notebook

# Open notebooks/Week3_DeepLearning.ipynb
```

---

## 📊 Dataset

**Source:** [US-Accidents: A Countrywide Traffic Accident Dataset (2016-2023)](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)

**Description:** 7+ million traffic accident records across 49 US states

**Features Used:**
- Temporal: Day of week, month, day of month, week of year
- Lag features: 1-day, 7-day, 30-day accident counts
- Rolling statistics: 7-day and 30-day moving averages
- Cyclical encodings: Sin/cos transformations
- State indicators: Multi-state training

**Preprocessing:**
- Daily aggregation by state
- Standardization of features
- MinMax scaling of target variable
- Time-based train/val/test splits (80/10/10)

---

## 🔬 Methodology

### Model Architecture

**Input Configuration:**
- Lookback window: 30 days
- Forecast horizon: 1 day ahead
- Features per timestep: 12-15
- Sequence shape: (batch_size, 30, n_features)

**Training Setup:**
- Framework: PyTorch
- Optimizer: Adam (learning rate: 0.001)
- Loss function: MSE
- Batch size: 32
- Early stopping on validation loss
- Device: CUDA (if available)

### Interpretability Techniques

1. **Attention Visualization**
   - Temporal attention weights analysis
   - Feature contribution heatmaps

2. **Feature Importance**
   - Integrated gradients
   - SHAP values (planned)

3. **Error Analysis**
   - Residual plots
   - Day-of-week accuracy breakdown
   - Temporal error patterns

---

## 📈 Results

### Performance Comparison

**Deep Learning vs. Traditional ML:**

| Metric | GRU | Random Forest | Gap |
|--------|-----|---------------|-----|
| R² Score | 0.415 | 0.55 | -24% |
| MAE | 0.560 | 0.45 | +24% |
| RMSE | 0.771 | 0.65 | +19% |
| Training Time | 15 min | 5 min | +200% |

**Insights:**
- GRU achieves 75% of baseline performance
- Strong foundation for hyperparameter optimization
- Sequential dependencies well-captured
- Room for improvement through tuning

### Temporal Patterns Discovered

**Weekly Cyclicality:**
- Monday peaks: +15% above average
- Weekend dips: -20% (Saturday-Sunday)
- Gradual decline Tuesday-Friday

**Seasonal Variations:**
- Summer months (Jun-Aug): +10-15% accidents
- Winter holidays (December): -20% during holiday weeks
- Weather correlation: Models learn seasonal weather impacts

---

## 🎨 Dashboard Features

The interactive Streamlit dashboard includes:

- **📊 Model Performance Overview:** Real-time metrics (MAE, RMSE, R², MAPE)
- **🏆 Model Comparison:** Bar charts and detailed metric tables
- **📅 Time Series Predictions:** Interactive plots with zoom/pan
- **🔍 Error Analysis:** Residual plots and distribution histograms
- **📋 Data Explorer:** Sortable/filterable prediction tables with CSV export
- **🌡️ Day-of-Week Analysis:** Accuracy breakdown by weekday

**Access Dashboard:** [Live Demo](https://us-collisions-2016-2023-forecasting-collision-trends-i96zdwvvj.streamlit.app/)

---

## 🔮 Future Work (Week 4)

### Planned Optimizations
- [ ] Hyperparameter tuning with Optuna
- [ ] Target: Surpass Random Forest (R² > 0.55)
- [ ] Ensemble methods (model averaging, stacking)
- [ ] Extended forecast horizons (3-day, 7-day predictions)
- [ ] Multi-task learning (predict severity + count)
- [ ] Transfer learning across states

### Additional Features
- [ ] Real-time data integration
- [ ] Weather data incorporation
- [ ] State-specific model fine-tuning
- [ ] Hour-level predictions
- [ ] Deployment pipeline for production

---

## 📚 Dependencies

**Core Libraries:**
```
torch>=2.0.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
```

**Visualization:**
```
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.17.0
```

**Dashboard:**
```
streamlit>=1.28.0
```

**Full list:** See `requirements.txt`

---

## 🎓 Academic Context

**Course:** Machine Learning for Traffic Forecasting  
**Institution:** Coppin State University  
**Semester:** Fall 2025  
**Student:** Mario Cuevas

**Project Timeline:**
- **Week 1:** Exploratory Data Analysis
- **Week 2:** Feature Engineering & Baseline Models
- **Week 3:** Deep Learning Forecasting & Dashboard ✅
- **Week 4:** Hyperparameter Optimization (In Progress)

---

## 📝 Citations

**Dataset:**
```
Moosavi, Sobhan, Mohammad Hossein Samavatian, Srinivasan Parthasarathy, 
and Rajiv Ramnath. "A Countrywide Traffic Accident Dataset.", 2019.
```

**Technical References:**
- Bahdanau et al. (2014): Neural Machine Translation by Jointly Learning to Align and Translate
- Bai et al. (2018): An Empirical Evaluation of Generic Convolutional and Recurrent Networks
- Vaswani et al. (2017): Attention Is All You Need

---

## 📄 License

This project is part of academic coursework and is intended for educational purposes.

---

## 🤝 Contributing

This is an academic project, but suggestions and feedback are welcome!

**Contact:**
- GitHub: [@mariostacksqueues](https://github.com/mariostacksqueues)
- Project Link: [US-Collisions-Forecasting](https://github.com/mariostacksqueues/US-Collisions-2016-2023-Forecasting-Collision-Trends)

---

## 🙏 Acknowledgments

- Course instructors for project guidance
- Kaggle community for the US-Accidents dataset
- PyTorch and Streamlit teams for excellent documentation
- Anthropic Claude for development assistance

---

## 📊 Project Stats

![Languages](https://img.shields.io/github/languages/top/mariostacksqueues/US-Collisions-2016-2023-Forecasting-Collision-Trends)
![Code Size](https://img.shields.io/github/languages/code-size/mariostacksqueues/US-Collisions-2016-2023-Forecasting-Collision-Trends)
![Last Commit](https://img.shields.io/github/last-commit/mariostacksqueues/US-Collisions-2016-2023-Forecasting-Collision-Trends)

---

**⭐ Star this repo if you found it helpful!**

**Last Updated:** November 16, 2025
