"""
Week 3: Interactive Traffic Accident Forecasting Dashboard
US Collisions Deep Learning Visualization
Mario Cuevas - November 2025

Run with: streamlit run streamlit_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="US Traffic Accident Forecasting",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    h1 {
        color: #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# LOAD DATA
# ============================================================================

@st.cache_data
def load_data():
    """Load model predictions and results"""
    try:
        predictions = pd.read_csv('results/model_predictions.csv')
        
        start_date = datetime(2023, 1, 1)  # Adjust based on your data
        predictions['Date'] = [start_date + timedelta(days=i) for i in range(len(predictions))]
        
        column_mapping = {
            'LSTM_Pred': 'LSTM + Attention',
            'GRU_Pred': 'GRU',
            'TCN_Pred': 'TCN',
            'Transformer_Pred': 'Transformer',
            'Actual': 'Actual'
        }
        predictions = predictions.rename(columns=column_mapping)
        
        comparison = pd.read_csv('results/model_comparison_results.csv')
        
        return predictions, comparison
    except FileNotFoundError as e:
        st.error(f"Data files not found: {e}")
        st.info("Please ensure the following files exist:")
        st.code("""
        results/model_predictions.csv
        results/model_comparison_results.csv
        """)
        return None, None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Check that your CSV files have the correct format")
        return None, None

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.title("🚗 US Traffic Accident Forecasting Dashboard")
    st.markdown("### Deep Learning Model Analysis & Predictions")
    st.markdown("---")
    
    predictions, comparison = load_data()
    if predictions is None:
        st.stop()
    
    with st.sidebar:
        st.markdown("## 📊 Dashboard Controls")
        
        st.markdown("### Model Selection")
        available_models = [col for col in predictions.columns if col not in ['Date', 'Actual']]
        selected_model = st.selectbox(
            "Choose Model",
            available_models,
            index=1  # Default to GRU
        )
        
        st.markdown("### Filters")
        show_all_data = st.checkbox("Show all available data", value=True)
        
        if not show_all_data:
            min_date = predictions['Date'].min()
            max_date = predictions['Date'].max()
            
            date_range = st.slider(
                "Select Date Range",
                min_value=min_date.to_pydatetime(),
                max_value=max_date.to_pydatetime(),
                value=(min_date.to_pydatetime(), max_date.to_pydatetime()),
                format="YYYY-MM-DD"
            )
        else:
            date_range = (predictions['Date'].min(), predictions['Date'].max())
        
        st.markdown("### About")
        st.info("""
        This dashboard visualizes deep learning model predictions for US traffic accidents.
        
        **Models:**
        - GRU (Best: R² = 0.415)
        - LSTM + Attention
        - TCN
        - Transformer
        
        **Data:** 2016-2023 US Accidents
        """)
    
    # Filter data by date range
    mask = (predictions['Date'] >= date_range[0]) & (predictions['Date'] <= date_range[1])
    filtered_predictions = predictions[mask]
    
    # ========================================================================
    # SECTION 1: KEY METRICS
    # ========================================================================
    
    st.markdown("## 📈 Model Performance Overview")
    
    cols = st.columns(4)
    
    # Calculate metrics for selected model
    if 'Actual' in filtered_predictions.columns and selected_model in filtered_predictions.columns:
        actual = filtered_predictions['Actual'].values
        predicted = filtered_predictions[selected_model].values
        
        mae = np.mean(np.abs(actual - predicted))
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        mape = np.mean(np.abs((actual - predicted) / (actual + 1e-10))) * 100
        r2 = 1 - (np.sum((actual - predicted) ** 2) / np.sum((actual - np.mean(actual)) ** 2))
        
        with cols[0]:
            st.metric("Mean Absolute Error", f"{mae:.3f}")
        with cols[1]:
            st.metric("Root Mean Squared Error", f"{rmse:.3f}")
        with cols[2]:
            st.metric("R² Score", f"{r2:.3f}")
        with cols[3]:
            st.metric("MAPE", f"{mape:.2f}%")
    
    st.markdown("---")
    
    # ========================================================================
    # SECTION 2: MODEL COMPARISON
    # ========================================================================
    
    st.markdown("## 🏆 Model Comparison")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Bar chart comparison
        fig_comparison = go.Figure()
        
        fig_comparison.add_trace(go.Bar(
            x=comparison['Model'],
            y=comparison['R²'],
            name='R² Score',
            marker_color='#1f77b4',
            text=comparison['R²'].round(3),
            textposition='outside'
        ))
        
        fig_comparison.update_layout(
            title="Model Performance (R² Score)",
            xaxis_title="Model",
            yaxis_title="R² Score",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig_comparison, use_container_width=True)
    
    with col2:
        # Table comparison
        st.markdown("### Detailed Metrics")
        comparison_display = comparison.copy()
        comparison_display['MAE'] = comparison_display['MAE'].round(3)
        comparison_display['RMSE'] = comparison_display['RMSE'].round(3)
        comparison_display['R²'] = comparison_display['R²'].round(3)
        
        st.dataframe(
            comparison_display,
            hide_index=True,
            use_container_width=True,
            height=200
        )
        
        # Highlight best model
        best_model = comparison.loc[comparison['R²'].idxmax(), 'Model']
        st.success(f"🌟 Best Model: **{best_model}**")
    
    st.markdown("---")
    
    # ========================================================================
    # SECTION 3: TIME SERIES PREDICTIONS
    # ========================================================================
    
    st.markdown("## 📅 Time Series Predictions")
    
    # Time series plot
    fig_timeseries = go.Figure()
    
    # Actual values
    fig_timeseries.add_trace(go.Scatter(
        x=filtered_predictions['Date'],
        y=filtered_predictions['Actual'],
        mode='lines',
        name='Actual',
        line=dict(color='black', width=2)
    ))
    
    # Predicted values
    fig_timeseries.add_trace(go.Scatter(
        x=filtered_predictions['Date'],
        y=filtered_predictions[selected_model],
        mode='lines',
        name=f'{selected_model} Prediction',
        line=dict(color='#1f77b4', width=2, dash='dot')
    ))
    
    fig_timeseries.update_layout(
        title=f"Accident Count Predictions - {selected_model}",
        xaxis_title="Date",
        yaxis_title="Accident Count (Normalized)",
        height=500,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig_timeseries, use_container_width=True)
    
    # ========================================================================
    # SECTION 4: ERROR ANALYSIS
    # ========================================================================
    
    st.markdown("## 🔍 Prediction Error Analysis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Residual plot
        residuals = filtered_predictions['Actual'] - filtered_predictions[selected_model]
        
        fig_residuals = go.Figure()
        
        fig_residuals.add_trace(go.Scatter(
            x=filtered_predictions['Date'],
            y=residuals,
            mode='markers',
            name='Residuals',
            marker=dict(color='#ff7f0e', size=5, opacity=0.6)
        ))
        
        fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
        
        fig_residuals.update_layout(
            title="Prediction Residuals Over Time",
            xaxis_title="Date",
            yaxis_title="Residual (Actual - Predicted)",
            height=400
        )
        
        st.plotly_chart(fig_residuals, use_container_width=True)
    
    with col2:
        fig_hist = go.Figure()
        
        fig_hist.add_trace(go.Histogram(
            x=residuals,
            nbinsx=30,
            name='Residuals',
            marker_color='#2ca02c'
        ))
        
        fig_hist.update_layout(
            title="Residual Distribution",
            xaxis_title="Residual Value",
            yaxis_title="Frequency",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # ========================================================================
    # SECTION 5: PREDICTION ACCURACY BY DAY OF WEEK
    # ========================================================================
    
    st.markdown("## 📊 Accuracy Analysis by Day of Week")
    
    filtered_predictions['DayOfWeek'] = filtered_predictions['Date'].dt.day_name()
    
    dow_mae = []
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    for day in days:
        day_data = filtered_predictions[filtered_predictions['DayOfWeek'] == day]
        if len(day_data) > 0:
            mae = np.mean(np.abs(day_data['Actual'] - day_data[selected_model]))
            dow_mae.append(mae)
        else:
            dow_mae.append(0)
    
    fig_dow = go.Figure()
    
    fig_dow.add_trace(go.Bar(
        x=days,
        y=dow_mae,
        marker_color=['#d62728' if mae == max(dow_mae) else '#1f77b4' for mae in dow_mae],
        text=[f"{mae:.3f}" for mae in dow_mae],
        textposition='outside'
    ))
    
    fig_dow.update_layout(
        title="Prediction Error by Day of Week",
        xaxis_title="Day of Week",
        yaxis_title="Mean Absolute Error",
        height=400
    )
    
    st.plotly_chart(fig_dow, use_container_width=True)
    
    # ========================================================================
    # SECTION 6: INTERACTIVE DATA TABLE
    # ========================================================================
    
    st.markdown("## 📋 Detailed Predictions")
    
    display_data = filtered_predictions.copy()
    display_data['Error'] = (display_data['Actual'] - display_data[selected_model]).round(3)
    display_data['Error %'] = ((display_data['Error'] / (display_data['Actual'] + 1e-10)) * 100).round(2)
    
    display_columns = ['Date', 'Actual', selected_model, 'Error', 'Error %', 'DayOfWeek']
    display_data = display_data[display_columns]
    
    n_rows = st.slider("Number of rows to display", 10, 100, 20)
    
    st.dataframe(
        display_data.head(n_rows),
        hide_index=True,
        use_container_width=True
    )
    
    csv = display_data.to_csv(index=False)
    st.download_button(
        label="📥 Download Full Predictions",
        data=csv,
        file_name=f"{selected_model}_predictions.csv",
        mime="text/csv"
    )
    
    # ========================================================================
    # SECTION 7: MODEL INSIGHTS
    # ========================================================================
    
    st.markdown("---")
    st.markdown("## 💡 Key Insights")
    
    col1, col2, col3 = st.columns(3)
    
    best_model = comparison.loc[comparison['R²'].idxmax(), 'Model']
    best_r2 = comparison['R²'].max()
    
    with col1:
        st.markdown("### 🎯 Best Performing Model")
        st.write(f"**{best_model}** achieves the highest R² score of {best_r2:.3f}")
        st.write("GRU architecture provides the best balance of accuracy and computational efficiency.")
    
    with col2:
        st.markdown("### 📈 Temporal Patterns")
        st.write("Models successfully capture:")
        st.write("- Weekly cycles (weekday vs weekend)")
        st.write("- Recent trend dependencies")
        st.write("- Seasonal variations")
    
    with col3:
        st.markdown("### 🔮 Next Steps")
        st.write("Week 4 optimization will:")
        st.write("- Tune hyperparameters with Optuna")
        st.write("- Target R² > 0.55")
        st.write("- Ensemble multiple models")
    
    # ========================================================================
    # FOOTER
    # ========================================================================
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>US Traffic Accident Forecasting Dashboard | Week 3 Deep Learning Project</p>
        <p>Mario Cuevas | November 2025 | Machine Learning Course</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()