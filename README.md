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

- **Source:** [US Accidents (2016-2023) - Kaggle](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)
- **Size:** 7.7M accident records, 49 states
- **Processed:** ~2,500 daily observations, 22 engineered features

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run notebook in Google Colab or Jupyter
jupyter notebook notebooks/Week3_DeepLearning.ipynb
```

## 🔬 Methodology

### Week 2: Baselines
- Random Forest: R² = 0.55 ✓
- Prophet: R² = 0.14

### Week 3: Deep Learning
**Quick Fixes Applied:**
1. Forecast horizon: 7 days → 1 day
2. Lookback window: 30 days → 14 days  
3. Epochs: 50 → 100
4. Simplified architecture

**Results:** GRU reached 75% of RF baseline!

## 🛠️ Tech Stack

Python 3.8+ | TensorFlow | Scikit-learn | Pandas | Plotly

## 📁 Structure

```
├── notebooks/          # Jupyter notebooks
├── models/             # Saved models  
├── results/            # Performance metrics & visualizations
├── reports/            # Documentation
└── requirements.txt    # Dependencies
```

## 🎯 Next Steps (Week 4)

- [ ] Hyperparameter optimization with Optuna
- [ ] Target: R² > 0.60 (beat baseline!)
- [ ] Feature ablation studies

## 👤 Author

**Mario Cuevas** - ML Coursework Project

## 📄 License

MIT License

---

⭐ Star if helpful! | 🚧 Status: Week 3 Complete, Week 4 In Progress
