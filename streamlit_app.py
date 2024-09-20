import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf

# Set the title and favicon that appear in the browser's tab bar.
st.set_page_config(
    page_title='Monte Carlo Stock Price Simulation',
    page_icon=':chart_with_upwards_trend:',  # Stock chart emoji.
)

# -------------------------------------------------------------------
# Declare some useful functions.

def geo_paths(S, T, r, q, sigma, steps, N):
    """Generates paths for a geometric Brownian motion."""
    dt = T / steps
    ST = np.log(S) + np.cumsum(((r - q - sigma**2 / 2) * dt + 
                               sigma * np.sqrt(dt) * 
                               np.random.normal(size=(steps, N))), axis=0)
    return np.exp(ST)

@st.cache_data
def get_stock_data(ticker, start, end):
    """Fetch stock data using yfinance."""
    stock_data = yf.download(ticker, start=start, end=end)
    return stock_data['Close']

# -------------------------------------------------------------------
# Page content and user interaction

# Set the title that appears at the top of the page.
'''
# :chart_with_upwards_trend: Monte Carlo Stock Price Simulation

Simulate future stock prices using the Monte Carlo method. Select a stock, and adjust parameters like the risk-free rate, volatility, and time horizon.
'''

# Sidebar input section for parameters
st.sidebar.header("Monte Carlo Simulation Parameters")

stock_ticker = st.sidebar.text_input('Enter stock ticker (e.g., AAPL, TSLA, MSFT):', 'AAPL')

# Date range input for historical data fetching
end_date = dt.datetime.now()
start_date = end_date - dt.timedelta(days=300)

# Fetch stock data
if stock_ticker:
    stock_data = get_stock_data(stock_ticker, start=start_date, end=end_date)
    st.write(f"Displaying closing prices for {stock_ticker}:")
    st.line_chart(stock_data)

    # Automatically set S0 (Initial Stock Price) to the most recent stock price
    S0 = stock_data[-1]
    st.sidebar.write(f"Latest Stock Price (S_0): {S0:.2f}")

    # Sidebar sliders for other parameters
    K = st.sidebar.slider('Strike Price (K)', min_value=50, max_value=500, value=int(S0 * 1.1))
    r = st.sidebar.slider('Risk-Free Rate (r)', min_value=0.0, max_value=0.1, value=0.05, step=0.01)
    sigma = st.sidebar.slider('Volatility (σ)', min_value=0.1, max_value=1.0, value=0.2, step=0.01)
    T = st.sidebar.slider('Time to Maturity (T)', min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    N = st.sidebar.slider('Number of Simulations (N)', min_value=10, max_value=1000, value=100)

    # Time steps fixed at 100 for now
    steps = 100

    # Perform Monte Carlo simulation
    paths = geo_paths(S0, T, r, 0, sigma, steps, N)

    # Plot the simulation
    st.subheader('Monte Carlo Simulation Results')
    fig, ax = plt.subplots()
    ax.plot(paths)
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Stock Price")
    ax.set_title(f"Simulated Stock Price Paths for {stock_ticker}")
    st.pyplot(fig)

    # Displaying some statistics
    st.write(f"Simulated final stock price mean: {paths[-1].mean():.2f}")
    st.write(f"Simulated final stock price standard deviation: {paths[-1].std():.2f}")

else:
    st.warning("Please enter a valid stock ticker.")
