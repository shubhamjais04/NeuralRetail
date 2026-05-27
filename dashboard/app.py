# ============================================================
# NeuralRetail — Streamlit Dashboard
# Amdox Technologies | Data Science & Analytics
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PAGE CONFIGURATION — must be first Streamlit command
# ============================================================
st.set_page_config(
    page_title="NeuralRetail | Amdox Technologies",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS — Amdox brand colors
# ============================================================
st.markdown("""
<style>
    /* Main background */
    .main { background-color: #0f0f0f; }

    /* Sidebar */
    .css-1d391kg { background-color: #1a1a1a; }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1a1a, #2a2a2a);
        border: 1px solid #E84E1B;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 5px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #E84E1B;
    }
    .metric-label {
        font-size: 13px;
        color: #aaaaaa;
        margin-top: 5px;
    }

    /* Headers */
    h1, h2, h3 { color: #E84E1B !important; }

    /* Streamlit metric */
    [data-testid="stMetricValue"] {
        color: #E84E1B;
        font-size: 24px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA LOADING — cached so it only loads once
# ============================================================
@st.cache_data
def load_data():
    """Load all datasets needed for the dashboard"""
    base = os.path.dirname(os.path.abspath(__file__))

    # Main clean dataset
    df = pd.read_csv(
        os.path.join(base, '../data/online_retail_clean.csv'),
        parse_dates=['InvoiceDate']
    )

    # RFM features
    rfm = pd.read_csv(os.path.join(base, '../data/rfm_features.csv'))

    # Daily sales
    daily = pd.read_csv(
        os.path.join(base, '../data/daily_sales_features.csv'),
        parse_dates=['Date']
    )

    # Forecast results
    forecast = pd.read_csv(
        os.path.join(base, '../data/forecast_results.csv'),
        parse_dates=['Date']
    )

    return df, rfm, daily, forecast

@st.cache_resource
def load_models():
    """Load trained ML models"""
    base = os.path.dirname(os.path.abspath(__file__))
    churn_model    = joblib.load(os.path.join(base, '../src/models/churn_model.pkl'))
    churn_features = joblib.load(os.path.join(base, '../src/models/churn_features.pkl'))
    return churn_model, churn_features

# Load everything
df, rfm, daily, forecast = load_data()
churn_model, churn_features = load_models()

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
st.sidebar.image("https://via.placeholder.com/200x60/E84E1B/FFFFFF?text=AMDOX", width=200)
st.sidebar.title("🛍️ NeuralRetail")
st.sidebar.markdown("**AI Sales Intelligence Platform**")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate to:",
    [
        "📊 Executive Overview",
        "📈 Demand Forecast",
        "👥 Customer Intelligence",
        "⚠️ Churn Risk Analysis",
        "🧪 Model Tracker"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Data Range:**")
st.sidebar.info(f"{df['InvoiceDate'].min().date()} → {df['InvoiceDate'].max().date()}")
st.sidebar.markdown("**Built with ❤️ by Amdox DS Team**")

# ============================================================
# PAGE 1 — EXECUTIVE OVERVIEW
# ============================================================
if page == "📊 Executive Overview":
    st.title("📊 Executive Overview")
    st.markdown("**Key Performance Indicators — NeuralRetail Platform**")
    st.markdown("---")

    # KPI Metrics Row
    col1, col2, col3, col4, col5 = st.columns(5)

    total_revenue   = df['TotalPrice'].sum()
    total_orders    = df['InvoiceNo'].nunique()
    total_customers = df['CustomerID'].nunique()
    total_products  = df['StockCode'].nunique()
    avg_order_value = total_revenue / total_orders

    col1.metric("💰 Total Revenue",    f"£{total_revenue:,.0f}")
    col2.metric("🛒 Total Orders",     f"{total_orders:,}")
    col3.metric("👥 Customers",        f"{total_customers:,}")
    col4.metric("📦 Products",         f"{total_products:,}")
    col5.metric("💳 Avg Order Value",  f"£{avg_order_value:,.2f}")

    st.markdown("---")

    # Monthly Revenue Chart
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("📈 Monthly Revenue Trend")
        df['YearMonth'] = df['InvoiceDate'].dt.to_period('M').astype(str)
        monthly = df.groupby('YearMonth')['TotalPrice'].sum().reset_index()
        fig = px.line(
            monthly, x='YearMonth', y='TotalPrice',
            markers=True,
            color_discrete_sequence=['#E84E1B'],
            labels={'TotalPrice': 'Revenue (£)', 'YearMonth': 'Month'}
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis_tickangle=-45,
            font_color='white'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("🌍 Revenue by Country")
        country_rev = df.groupby('Country')['TotalPrice'].sum().reset_index()
        country_rev = country_rev.sort_values('TotalPrice', ascending=False).head(10)
        fig2 = px.bar(
            country_rev, x='TotalPrice', y='Country',
            orientation='h',
            color='TotalPrice',
            color_continuous_scale=['#FBBA13', '#F7941D', '#E84E1B'],
            labels={'TotalPrice': 'Revenue (£)', 'Country': ''}
        )
        fig2.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            yaxis={'categoryorder': 'total ascending'},
            showlegend=False,
            coloraxis_showscale=False
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Top Products
    st.subheader("🏆 Top 10 Products by Revenue")
    top_products = df.groupby('Description')['TotalPrice'].sum().reset_index()
    top_products = top_products.sort_values('TotalPrice', ascending=False).head(10)
    fig3 = px.bar(
        top_products, x='Description', y='TotalPrice',
        color='TotalPrice',
        color_continuous_scale=['#FBBA13', '#F7941D', '#E84E1B'],
        labels={'TotalPrice': 'Revenue (£)', 'Description': 'Product'}
    )
    fig3.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        xaxis_tickangle=-45,
        coloraxis_showscale=False
    )
    st.plotly_chart(fig3, use_container_width=True)

# ============================================================
# PAGE 2 — DEMAND FORECAST
# ============================================================
elif page == "📈 Demand Forecast":
    st.title("📈 Demand Forecast Explorer")
    st.markdown("**30-Day Revenue Forecast using Machine Learning**")
    st.markdown("---")

    # Actual vs Predicted
    st.subheader("🔮 Actual vs Predicted Revenue")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=forecast['Date'], y=forecast['Actual'],
        name='Actual Revenue',
        line=dict(color='#E84E1B', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=forecast['Date'], y=forecast['Predicted'],
        name='Predicted Revenue',
        line=dict(color='#F7941D', width=2, dash='dash')
    ))
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        hovermode='x unified',
        xaxis_title='Date',
        yaxis_title='Revenue (£)'
    )
    st.plotly_chart(fig, use_container_width=True)

    # MAPE metric
    mape = np.mean(np.abs(
        (forecast['Actual'] - forecast['Predicted']) / forecast['Actual']
    )) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("📉 MAPE", f"{mape:.2f}%", help="Mean Absolute Percentage Error — lower is better")
    col2.metric("🎯 Target MAPE", "≤ 10%")
    col3.metric("✅ Status", "On Target" if mape <= 10 else "Above Target")

    st.markdown("---")

    # Daily sales trend
    st.subheader("📊 Historical Daily Revenue")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=daily['Date'], y=daily['Revenue'],
        name='Daily Revenue',
        line=dict(color='#E84E1B', width=1),
        fill='tozeroy',
        fillcolor='rgba(232,78,27,0.1)'
    ))
    fig2.add_trace(go.Scatter(
        x=daily['Date'], y=daily['Rolling_Mean_7'],
        name='7-Day Avg',
        line=dict(color='#F7941D', width=2)
    ))
    fig2.add_trace(go.Scatter(
        x=daily['Date'], y=daily['Rolling_Mean_30'],
        name='30-Day Avg',
        line=dict(color='#FBBA13', width=2)
    ))
    fig2.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        hovermode='x unified',
        xaxis_title='Date',
        yaxis_title='Revenue (£)'
    )
    st.plotly_chart(fig2, use_container_width=True)

# ============================================================
# PAGE 3 — CUSTOMER INTELLIGENCE
# ============================================================
elif page == "👥 Customer Intelligence":
    st.title("👥 Customer Intelligence")
    st.markdown("**RFM Segmentation — Know Your Customers**")
    st.markdown("---")

    # Segment overview
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Customer Segments")
        seg_counts = rfm['Segment'].value_counts().reset_index()
        seg_counts.columns = ['Segment', 'Count']
        fig = px.pie(
            seg_counts, values='Count', names='Segment',
            color_discrete_sequence=['#E84E1B','#F7941D','#FBBA13',
                                     '#27AE60','#2980B9','#8E44AD'],
            hole=0.4
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("💰 Revenue by Segment")
        seg_rev = rfm.groupby('Segment')['Monetary'].sum().reset_index()
        seg_rev = seg_rev.sort_values('Monetary', ascending=False)
        fig2 = px.bar(
            seg_rev, x='Segment', y='Monetary',
            color='Monetary',
            color_continuous_scale=['#FBBA13','#F7941D','#E84E1B'],
            labels={'Monetary': 'Total Revenue (£)', 'Segment': ''}
        )
        fig2.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            coloraxis_showscale=False,
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig2, use_container_width=True)

    # RFM Distribution
    st.subheader("📈 RFM Score Distributions")
    col3, col4, col5 = st.columns(3)

    with col3:
        fig3 = px.histogram(
            rfm, x='Recency', nbins=30,
            title='Recency Distribution',
            color_discrete_sequence=['#E84E1B']
        )
        fig3.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        fig4 = px.histogram(
            rfm, x='Frequency', nbins=30,
            title='Frequency Distribution',
            color_discrete_sequence=['#F7941D']
        )
        fig4.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig4, use_container_width=True)

    with col5:
        fig5 = px.histogram(
            rfm, x='Monetary', nbins=30,
            title='Monetary Distribution',
            color_discrete_sequence=['#FBBA13']
        )
        fig5.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig5, use_container_width=True)

    # Customer table
    st.subheader("🔍 Customer Details")
    segment_filter = st.selectbox(
        "Filter by Segment:",
        ['All'] + list(rfm['Segment'].unique())
    )
    if segment_filter == 'All':
        display_rfm = rfm
    else:
        display_rfm = rfm[rfm['Segment'] == segment_filter]

    st.dataframe(
        display_rfm[['CustomerID','Recency','Frequency',
                      'Monetary','RFM_Score','Segment','Churned']]\
        .sort_values('Monetary', ascending=False)\
        .reset_index(drop=True),
        use_container_width=True,
        height=400
    )

# ============================================================
# PAGE 4 — CHURN RISK ANALYSIS
# ============================================================
elif page == "⚠️ Churn Risk Analysis":
    st.title("⚠️ Churn Risk Analysis")
    st.markdown("**Identify customers at risk of leaving**")
    st.markdown("---")

    # Churn overview metrics
    churn_rate = rfm['Churned'].mean() * 100
    churned    = rfm['Churned'].sum()
    active     = len(rfm) - churned

    col1, col2, col3 = st.columns(3)
    col1.metric("📊 Overall Churn Rate", f"{churn_rate:.1f}%")
    col2.metric("⚠️ Churned Customers",  f"{churned:,}")
    col3.metric("✅ Active Customers",   f"{active:,}")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("🥧 Churn Distribution")
        churn_df = pd.DataFrame({
            'Status': ['Active', 'Churned'],
            'Count': [active, churned]
        })
        fig = px.pie(
            churn_df, values='Count', names='Status',
            color_discrete_sequence=['#27AE60', '#E84E1B'],
            hole=0.4
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("📊 Churn by Segment")
        churn_seg = rfm.groupby('Segment')['Churned'].mean().reset_index()
        churn_seg['Churn Rate %'] = churn_seg['Churned'] * 100
        churn_seg = churn_seg.sort_values('Churn Rate %', ascending=False)
        fig2 = px.bar(
            churn_seg, x='Segment', y='Churn Rate %',
            color='Churn Rate %',
            color_continuous_scale=['#27AE60', '#FBBA13', '#E84E1B'],
            labels={'Churn Rate %': 'Churn Rate (%)', 'Segment': ''}
        )
        fig2.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            coloraxis_showscale=False,
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig2, use_container_width=True)

    # SHAP feature importance image
    st.subheader("🔍 SHAP Feature Importance — Why Customers Churn")
    shap_img = '../evidently_reports/shap_importance.png'
    if os.path.exists(shap_img):
        st.image(shap_img, caption="SHAP Feature Importance", use_column_width=True)
    else:
        st.info("Run the model training notebook to generate SHAP chart")

    # High risk customers table
    st.subheader("🚨 High Risk Customers (Churned)")
    high_risk = rfm[rfm['Churned'] == 1].sort_values('Monetary', ascending=False)
    st.dataframe(
        high_risk[['CustomerID','Recency','Frequency',
                   'Monetary','RFM_Score','Segment']]\
        .reset_index(drop=True),
        use_container_width=True,
        height=350
    )

    # Download button
    csv = high_risk.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download High Risk Customer List",
        data=csv,
        file_name='high_risk_customers.csv',
        mime='text/csv'
    )

# ============================================================
# PAGE 5 — MODEL TRACKER
# ============================================================
elif page == "🧪 Model Tracker":
    st.title("🧪 Model Performance Tracker")
    st.markdown("**MLflow Experiment Tracking**")
    st.markdown("---")

    # Model performance summary
    st.subheader("📊 Trained Models Summary")

    models_data = pd.DataFrame({
        'Model': ['XGBoost Churn', 'LightGBM Churn', 'Linear Regression Demand'],
        'Task': ['Churn Prediction', 'Churn Prediction', 'Demand Forecasting'],
        'Primary Metric': ['AUC-ROC', 'AUC-ROC', 'MAPE'],
        'Target': ['≥ 0.90', '≥ 0.90', '≤ 10%'],
        'Status': ['✅ Trained', '✅ Trained', '✅ Trained']
    })
    st.dataframe(models_data, use_container_width=True)

    st.markdown("---")
    st.subheader("📁 Saved Model Files")

    model_files = [
        '../src/models/churn_model.pkl',
        '../src/models/churn_features.pkl',
        '../src/models/forecast_model.pkl',
        '../src/models/forecast_features.pkl'
    ]

    for f in model_files:
        if os.path.exists(f):
            size = os.path.getsize(f) / 1024
            st.success(f"✅ {os.path.basename(f)} ({size:.1f} KB)")
        else:
            st.error(f"❌ {os.path.basename(f)} — not found")

    st.markdown("---")
    st.subheader("🚀 Launch MLflow UI")
    st.markdown("Run this command in your VS Code terminal to view all experiments:")
    st.code("mlflow ui --backend-store-uri ../mlflow_runs", language="bash")
    st.info("Then open http://localhost:5000 in your browser")

    # Evidently reports
    st.markdown("---")
    st.subheader("📊 Generated Reports")
    report_files = os.listdir('../evidently_reports') \
        if os.path.exists('../evidently_reports') else []

    if report_files:
        for f in report_files:
            st.success(f"✅ {f}")
    else:
        st.info("No reports generated yet")