import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# mathematical engine
def calculate_shannon_entropy(binned_window):
    counts = np.bincount(binned_window.astype(int), minlength=11)[1:]
    total_counts = np.sum(counts)
    if total_counts == 0:
        return 0
    p=counts/total_counts
    p= p[p>0]

    entropy = -np.sum(p*np.log(p))
    normalized_entropy = entropy/np.log(10)

    return normalized_entropy

# 1. Config and Headers
st.set_page_config(page_title="Regime Monitor", page_icon="📈", layout="wide")
st.title("📈 Quantitative Market Regime Monitor")
st.markdown("---")

# 2. Sidebar Configuration Options
st.sidebar.header("Risk Engine Controls")

asset_choice = st.sidebar.selectbox(
    label="Select Asset Matrix", 
    options=["BTC-USD", "ETH-USD", "^GSPC"]
)

window_size = st.sidebar.slider(
    label="Rolling Window Size (Days)", 
    min_value=10, 
    max_value=200, 
    value=100
)

# 3. Dynamic Data Processing Engine
st.subheader(f"Live Historical Feed: {asset_choice}")

try:
    # Download raw price history
    raw_data = yf.download(asset_choice, start="2020-01-01")
    
    if not raw_data.empty:
        # Extract Close prices cleanly as a single Series to avoid MultiIndex issues
        close_prices = raw_data['Close']
        if isinstance(close_prices, pd.DataFrame):
            close_prices = close_prices.squeeze()

        # Build a clean 1D DataFrame
        data = pd.DataFrame({'Close': close_prices})
        
        # Render Closing Price Chart
        st.line_chart(data['Close'])
        
        # A. Calculate Log Returns
        data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
        data = data.dropna()
        
        # B. Quantile Binning (10 Bins)
        data['Binned'] = pd.qcut(data['Log_Returns'], q=10, labels=False, duplicates='drop') + 1
        
        # C. Rolling Shannon Entropy
        data['Entropy'] = data['Binned'].rolling(window=window_size).apply(calculate_shannon_entropy, raw=True)
        
        # D. Render the Entropy Wave Chart
        st.markdown("---")
        st.subheader(f"🌀 Normalized Rolling Shannon Entropy ({window_size}-Day Window)")
        st.line_chart(data['Entropy'])
        
        # E. Compute Rolling Volatility
        data['Volatility'] = data['Log_Returns'].rolling(window=window_size).std()
        st.markdown("---")
        st.subheader(f"📊 Rolling Volatility ({window_size}-Day Window)")
        st.line_chart(data['Volatility'])

        # F. Summary Metrics Display Cards
        st.markdown("---")
        st.subheader("💡 Market Regime Highlights")

        latest_entropy = data['Entropy'].iloc[-1]
        mean_entropy = data['Entropy'].mean()
        latest_vol = data['Volatility'].iloc[-1]

        col1, col2, col3 = st.columns(3)
        col1.metric("Latest Entropy", f"{latest_entropy:.3f}")
        col2.metric("Mean Market Entropy", f"{mean_entropy:.3f}")
        col3.metric("Latest Volatility", f"{latest_vol:.4f}")

    else:
        st.error("Data pipeline structurally empty frame returned.")
        
except Exception as e:
    st.error(f"Error executing data engine layer fetch: {e}")
        
# G. Entropy Shock & Volatility Risk Thresholds (Option B)
st.markdown("---")
st.subheader("🚨 Market Regime Risk Detector")

# 1. Calculate statistical thresholds (95th & 5th percentiles)
high_vol_threshold = data['Volatility'].quantile(0.95)
low_entropy_threshold = data['Entropy'].quantile(0.05)

# 2. Get the current status
current_entropy = data['Entropy'].iloc[-1]
current_vol = data['Volatility'].iloc[-1]

# 3. Define Risk States
if current_vol > high_vol_threshold and current_entropy < low_entropy_threshold:
     st.error("⚠️ **CRITICAL RISK REGIME:** High Volatility + Extreme Low Entropy (Market Compression/Shock imminent!)")
elif current_vol > high_vol_threshold:
    st.warning("⚡ **HIGH VOLATILITY REGIME:** Market is experiencing elevated price turbulence.")
elif current_entropy < low_entropy_threshold:
    st.warning("🔍 **LOW ENTROPY REGIME:** Market returns are highly concentrated / predictable.")
else:
    st.success("🟢 **NORMAL REGIME:** Market metrics are operating within standard historical bands.")

# Display threshold values for transparency
st.caption(f"Historical 95th Percentile Volatility Threshold: `{high_vol_threshold:.4f}` | Historical 5th Percentile Entropy Threshold: `{low_entropy_threshold:.3f}`")