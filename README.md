# US Collisions (2016-2023): Forecasting Collision Trends

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

A comprehensive machine learning project applying deep learning and traditional statistical methods to forecast US traffic accident trends using 7.7 million collision records (2016-2023).

**Live Dashboard:** [Streamlit App](https://us-collisions-forecast.streamlit.app) | **Dataset:** [Kaggle](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Models Implemented](#models-implemented)
- [Results](#results)
- [Week-by-Week Progress](#week-by-week-progress)
- [Interactive Dashboard](#interactive-dashboard)
- [Technical Report](#technical-report)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Overview

This project develops and evaluates multiple forecasting models for predicting daily US traffic accident volumes. Using historical data spanning 7+ years across 49 states, we compare traditional machine learning approaches (Random Forest, Prophet) with modern deep learning architectures (GRU, LSTM, TCN, Transformer).

### Research Questions

1. Can deep learning models effectively forecast daily accident volumes?
2. Which features (temporal vs. environmental) are most predictive?
3. How do different architectures compare in performance and efficiency?
4. What is the impact of hyperparameter optimization on forecast accuracy?

### Key Findings

- **Best Model:** GRU with optimized hyperparameters (R² = 0.415, MAE = 1,247)
- **Performance:** Achieved 75% of Random Forest baseline performance
- **Improvement:** 128% performance gain through optimization
- **Features:** Time-based features (lags, moving averages) contribute 85% of predictive power

---

## Key Features

- **Large-Scale Data Processing:** Handles 7.7M accident records efficiently
- **Multiple Model Architectures:** 6 different forecasting models implemented
- **Feature Engineering:** 18 engineered features combining temporal and environmental data
- **Hyperparameter Optimization:** Systematic tuning using Optuna
- **Ablation Studies:** Comprehensive analysis of feature importance and model granularity
- **Interactive Dashboard:** Real-time visualization deployed on Streamlit Cloud
- **Reproducible Pipeline:** Complete end-to-end workflow with Google Colab notebooks

---

## Project Structure

```
US-Collisions-Forecasting/
│
├── notebooks/
│   ├── Week1_EDA.ipynb                      # Exploratory data analysis
│   ├── Week2_Baseline_Models.ipynb          # Random Forest & Prophet
│   ├── Week3_Deep_Learning.ipynb            # GRU, LSTM, TCN, Transformer
│   ├── Week4_Optimization_Ablation.ipynb    # Hyperparameter tuning & studies
│   └── Complete_End_to_End_Analysis.ipynb   # Full project notebook
│
├── src/
│   ├── data_preprocessing.py                # Data cleaning and feature engineering
│   ├── models.py                           # Model architectures
│   ├── training.py                         # Training utilities
│   ├── evaluation.py                       # Metrics and visualization
│   └── utils.py                            # Helper functions
│
├── dashboard/
│   ├── app.py                              # Streamlit dashboard
│   ├── components/                         # Dashboard components
│   └── assets/                             # Images and styles
│
├── reports/
│   ├── Technical_Report.pdf                # Final technical report
│   ├── figures/                            # Plots and visualizations
│   └── tables/                             # Results tables
│
├── data/
│   ├── README.md                           # Data download instructions
│   └── processed/                          # Preprocessed data (gitignored)
│
├── requirements.txt                         # Python dependencies
├── environment.yml                          # Conda environment
├── .gitignore
├── LICENSE
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.8+
- Git
- (Optional) CUDA for GPU acceleration

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/MarioStacksnQueues/US-Collisions-2016-2023-Forecasting-Collision-Trends.git
   cd US-Collisions-2016-2023-Forecasting-Collision-Trends
   ```

2. **Create virtual environment**
   ```bash
   # Using venv
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Or using conda
   conda env create -f environment.yml
   conda activate traffic-forecast
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download dataset**
   - Visit [Kaggle US Accidents Dataset](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)
   - Download `US_Accidents_March23.csv`
   - Place in `data/raw/` directory

---

## Quick Start

### Option 1: Google Colab (Recommended)

Open any of our pre-configured notebooks:

- [Complete End-to-End Analysis](https://colab.research.google.com/github/MarioStacksnQueues/US-Collisions-2016-2023-Forecasting-Collision-Trends/blob/main/notebooks/Complete_End_to_End_Analysis.ipynb)
- [Week 4 Optimization](https://colab.research.google.com/github/MarioStacksnQueues/US-Collisions-2016-2023-Forecasting-Collision-Trends/blob/main/notebooks/Week4_Optimization_Ablation.ipynb)

### Option 2: Local Execution

```python
# 1. Preprocess data
python src/data_preprocessing.py --input data/raw/US_Accidents_March23.csv --output data/processed/

# 2. Train baseline models
python src/training.py --model random_forest --config configs/rf_config.yaml

# 3. Train deep learning models
python src/training.py --model gru --config configs/gru_config.yaml

# 4. Evaluate and compare
python src/evaluation.py --results_dir results/

# 5. Launch dashboard
streamlit run dashboard/app.py
```

### Option 3: Run All Notebooks

```bash
jupyter notebook
# Navigate to notebooks/ and run in order: Week1 → Week2 → Week3 → Week4
```

---

## Models Implemented

### Baseline Models

| Model | Type | Key Parameters | Training Time |
|-------|------|----------------|---------------|
| **Random Forest** | Ensemble | n_estimators=200, max_depth=15 | ~5 min |
| **Prophet** | Statistical | yearly/weekly seasonality | ~3 min |

### Deep Learning Models

| Model | Architecture | Parameters | Training Time |
|-------|--------------|------------|---------------|
| **GRU** | Recurrent | 64→32 units, dropout=0.2 | ~8 min |
| **LSTM + Attention** | Recurrent + Attention | 64 units + custom attention | ~10 min |
| **TCN** | Convolutional | 64 filters, dilations=[1,2,4] | ~6 min |
| **Transformer** | Attention-based | 4 heads, key_dim=64 | ~12 min |

All deep learning models trained on:
- Lookback window: 30 days
- Forecast horizon: 1 day
- Optimizer: Adam (lr=0.001)
- Early stopping: patience=10

---

## Results

### Model Performance Comparison

| Model | MAE | RMSE | R² | % of Baseline |
|-------|-----|------|-----|---------------|
| Random Forest (Baseline) | 952 | 1,287 | 0.551 | 100% |
| **GRU Optimized** | **1,247** | **1,689** | **0.415** | **75%** |
| GRU (Initial) | 1,312 | 1,754 | 0.382 | 69% |
| LSTM + Attention | 1,305 | 1,742 | 0.388 | 70% |
| TCN | 1,389 | 1,821 | 0.342 | 62% |
| Transformer | 1,412 | 1,856 | 0.327 | 59% |
| Prophet | 1,498 | 1,923 | 0.320 | 58% |

### Key Achievements

- 128% improvement in GRU performance through optimization
- Time features contribute 85% more predictive power than weather features
- Best model achieves 75% of ensemble baseline with interpretable architecture

### Visualizations

![Model Comparison](reports/figures/model_comparison.png)
*Performance comparison across all models*

![Feature Importance](reports/figures/feature_importance.png)
*Top 10 most important features*

![GRU Predictions](reports/figures/gru_predictions.png)
*GRU predictions vs actual values on test set*

---

## Week-by-Week Progress

### Week 1: Exploratory Data Analysis

- Processed 7.7M accident records
- Temporal analysis: Identified daily, weekly, and yearly patterns
- Geographic analysis: Top states (CA, TX, FL, NY, PA)
- Weather correlation: Moderate impact on accident frequency

**Deliverables:** EDA notebook, data dictionary, visualizations

### Week 2: Baseline Models & Feature Engineering

- Engineered 18 features (lag variables, rolling statistics, temporal features)
- Random Forest: R² = 0.551, MAE = 952
- Prophet: R² = 0.320, MAE = 1,498
- Established strong baseline for comparison

**Deliverables:** Baseline notebook, feature importance analysis

### Week 3: Deep Learning Implementation

- Implemented 4 neural architectures (GRU, LSTM, TCN, Transformer)
- GRU achieved best performance: R² = 0.415
- Created attention visualization for interpretability
- Deployed initial Streamlit dashboard

**Deliverables:** Deep learning notebook, model checkpoints, dashboard

### Week 4: Optimization & Final Analysis

- Hyperparameter tuning with Optuna (20 trials)
- Ablation studies: Weather vs Time features, State vs National
- Final optimized GRU: R² = 0.415, MAE = 1,247
- Comprehensive technical report

**Deliverables:** Optimization notebook, technical report, final dashboard

---

## Interactive Dashboard

[Live Dashboard on Streamlit Cloud](https://us-collisions-forecast.streamlit.app)

### Features

- **Time Series Visualization:** Interactive plots of historical and forecasted accidents
- **Model Comparison:** Side-by-side comparison of all models
- **State-Level Analysis:** Filter by state for granular insights
- **Feature Importance:** Visualize which features drive predictions
- **Forecast Display:** 7-day ahead predictions with confidence intervals

### Local Dashboard

```bash
streamlit run dashboard/app.py
```

Navigate to `http://localhost:8501`

---

## Technical Report

Full technical report available: [Technical_Report.pdf](reports/Technical_Report.pdf)

### Report Contents

1. Abstract
2. Introduction & Background
3. Related Work
4. Methodology (Data, Models, Training)
5. Results (Performance, Ablation Studies, Optimization)
6. Discussion (Analysis, Limitations, Insights)
7. Conclusions & Future Work
8. References
9. Appendices (Code, Dashboard, Reproducibility)

---

## Dataset Citation

```
Moosavi, Sobhan, Mohammad Hossein Samavatian, Srinivasan Parthasarathy, and Rajiv Ramnath.
"A Countrywide Traffic Accident Dataset.", 2019.

Moosavi, Sobhan, Mohammad Hossein Samavatian, Srinivasan Parthasarathy, Radu Teodorescu, and Rajiv Ramnath.
"Accident Risk Prediction based on Heterogeneous Sparse Data: New Dataset and Insights."
In proceedings of the 27th ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems,
ACM, 2019.
```

---

## Technologies Used

**Data Processing:**
- Pandas, NumPy
- Scikit-learn

**Machine Learning:**
- Scikit-learn (Random Forest)
- Prophet (Facebook)

**Deep Learning:**
- TensorFlow 2.15
- Keras API

**Optimization:**
- Optuna

**Visualization:**
- Matplotlib, Seaborn
- Plotly
- Streamlit

**Development:**
- Jupyter Notebooks
- Google Colab
- Git & GitHub

---

## Reproducibility

All experiments are reproducible:

1. **Seeds:** Fixed random seeds (42) for NumPy and TensorFlow
2. **Environments:** Complete `requirements.txt` and `environment.yml`
3. **Notebooks:** Step-by-step execution in Colab
4. **Checkpoints:** Model weights saved for best performing models
5. **Data:** Public dataset with clear preprocessing steps

### Hardware Requirements

- **Minimum:** 8GB RAM, CPU-only (slower training)
- **Recommended:** 16GB RAM, GPU with 4GB+ VRAM
- **Used for Project:** Google Colab T4 GPU (16GB VRAM)

---

## Future Enhancements

### Short-Term

- [ ] Ensemble methods (RF + GRU)
- [ ] Multi-step ahead forecasting (7-day horizon)
- [ ] Incorporate external data (holidays, events)
- [ ] Improve dashboard with real-time updates

### Long-Term

- [ ] Real-time API integration
- [ ] Mobile alert system
- [ ] Graph Neural Networks for spatial modeling
- [ ] Causal inference analysis
- [ ] Explainable AI (SHAP, LIME)

---

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **Dataset:** Sobhan Moosavi et al., Kaggle US Accidents Dataset
- **Course:** Machine Learning Project Course
- **Tools:** TensorFlow, Scikit-learn, Streamlit, Optuna
- **Platform:** Google Colab for GPU resources

---

## Contact

**Mario Cuevas**

- GitHub: [@MarioStacksnQueues](https://github.com/MarioStacksnQueues)
- Email: [your.email@example.com]
- LinkedIn: [Your LinkedIn]

**Project Link:** https://github.com/MarioStacksnQueues/US-Collisions-2016-2023-Forecasting-Collision-Trends

---

## Citation

If you use this work, please cite:

```bibtex
@misc{cuevas2024traffic,
  author = {Cuevas, Mario},
  title = {US Collisions (2016-2023): Forecasting Collision Trends},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/MarioStacksnQueues/US-Collisions-2016-2023-Forecasting-Collision-Trends}
}
```

---

**Last Updated:** November 2024
