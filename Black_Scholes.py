import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import log, sqrt, exp  

#######################
# Page configuration
st.set_page_config(
    page_title="Black and Scholes Option P&L Model",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded")

# Define the Black-Scholes Model Class
class BlackScholes:
    def __init__(self, time_to_maturity, strike, current_price, volatility, interest_rate):
        self.time_to_maturity = time_to_maturity
        self.strike = strike
        self.current_price = current_price
        self.volatility = volatility
        self.interest_rate = interest_rate

    def calculate_prices(self):
        time_to_maturity = self.time_to_maturity
        strike = self.strike
        current_price = self.current_price
        volatility = self.volatility
        interest_rate = self.interest_rate

        d1 = (log(current_price / strike) + (interest_rate + 0.5 * volatility ** 2) * time_to_maturity) / (volatility * sqrt(time_to_maturity))
        d2 = d1 - volatility * sqrt(time_to_maturity)

        call_price = current_price * norm.cdf(d1) - (strike * exp(-(interest_rate * time_to_maturity)) * norm.cdf(d2))
        put_price = (strike * exp(-(interest_rate * time_to_maturity)) * norm.cdf(-d2)) - current_price * norm.cdf(-d1)

        self.call_price = call_price
        self.put_price = put_price

        return call_price, put_price

# Sidebar for User Inputs
with st.sidebar:
    st.title("📈 Black and Scholes P&L Model")
    st.markdown ("Austin Lochhead")
    
    current_price = st.number_input("Current Asset Price", value=100.0)
    strike = st.number_input("Strike Price", value=100.0)
    time_to_maturity = st.number_input("Time to Maturity (Years)", value=1.00)
    volatility = st.number_input("Volatility (σ)", value=0.20)
    interest_rate = st.number_input("Risk-Free Interest Rate", value=0.05)

    st.markdown("---")
    
    # User Inputs for Purchase Prices
    call_purchase_price = st.number_input("Purchase Price of Call", value=5.0)
    put_purchase_price = st.number_input("Purchase Price of Put", value=5.0)

    st.markdown("---")
    
    # Heatmap Parameters
    spot_min = st.number_input('Min Spot Price', min_value=0.01, value=current_price*0.8, step=0.01)
    spot_max = st.number_input('Max Spot Price', min_value=0.01, value=current_price*1.2, step=0.01)
    vol_min = st.slider('Min Volatility for Heatmap', min_value=0.01, max_value=1.0, value=volatility*0.5, step=0.01)
    vol_max = st.slider('Max Volatility for Heatmap', min_value=0.01, max_value=1.0, value=volatility*1.5, step=0.01)
    
    spot_range = np.linspace(spot_min, spot_max, 10)
    vol_range = np.linspace(vol_min, vol_max, 10)

# Function to Generate Profit & Loss Heatmaps
def plot_pnl_heatmap(bs_model, spot_range, vol_range, strike, call_purchase_price, put_purchase_price):
    call_pnl = np.zeros((len(vol_range), len(spot_range)))
    put_pnl = np.zeros((len(vol_range), len(spot_range)))
    
    for i, vol in enumerate(vol_range):
        for j, spot in enumerate(spot_range):
            bs_temp = BlackScholes(
                time_to_maturity=bs_model.time_to_maturity,
                strike=strike,
                current_price=spot,
                volatility=vol,
                interest_rate=bs_model.interest_rate
            )
            call_price, put_price = bs_temp.calculate_prices()
            
            # Calculate Profit & Loss
            call_pnl[i, j] = call_price - call_purchase_price
            put_pnl[i, j] = put_price - put_purchase_price

    # Call Option P&L Heatmap
    fig_call, ax_call = plt.subplots(figsize=(10, 8))
    sns.heatmap(call_pnl, xticklabels=np.round(spot_range, 2), yticklabels=np.round(vol_range, 2), annot=True, fmt=".2f", cmap="RdYlGn", ax=ax_call)
    ax_call.set_title('CALL OPTION P&L')
    ax_call.set_xlabel('Spot Price')
    ax_call.set_ylabel('Implied Volatility')

    # Put Option P&L Heatmap
    fig_put, ax_put = plt.subplots(figsize=(10, 8))
    sns.heatmap(put_pnl, xticklabels=np.round(spot_range, 2), yticklabels=np.round(vol_range, 2), annot=True, fmt=".2f", cmap="RdYlGn", ax=ax_put)
    ax_put.set_title('PUT OPTION P&L')
    ax_put.set_xlabel('Spot Price')
    ax_put.set_ylabel('Implied Volatility')
    
    return fig_call, fig_put

# Main Page for Output Display
st.title("Black and Scholes Option Profit & Loss Model")

# Display Table of Inputs with two decimal places
input_data = {"Current Asset Price ($)": [round(current_price, 2)],
    "Strike Price ($)": [round(strike, 2)],
    "Time to Maturity (Years)": [round(time_to_maturity, 2)],
    "Volatility (σ)": [round(volatility, 2)],
    "Risk-Free Interest Rate": [round(interest_rate, 2)],
    "Call Purchase Price ($)": [round(call_purchase_price, 2)],
    "Put Purchase Price ($)": [round(put_purchase_price, 2)]
}
input_df = pd.DataFrame(input_data)
st.table(input_df)


# Calculate Option Prices
bs_model = BlackScholes(time_to_maturity, strike, current_price, volatility, interest_rate)
call_price, put_price = bs_model.calculate_prices()

# Calculate Profit & Loss
call_pnl = call_price - call_purchase_price
put_pnl = put_price - put_purchase_price

# Display Profit & Loss
col1, col2 = st.columns(2)

with col1:
    st.metric(label="CALL P&L", value=f"${call_pnl:.2f}", delta=f"${call_pnl:.2f}")

with col2:
    st.metric(label="PUT P&L", value=f"${put_pnl:.2f}", delta=f"${put_pnl:.2f}")

st.markdown("---")

# Generate P&L Heatmaps
st.title("Interactive Heatmap - Profit & Loss for Call and Put Options")

col1, col2 = st.columns([1,1])

with col1:
    st.subheader("Call Option P&L Heatmap")
    heatmap_fig_call, _ = plot_pnl_heatmap(bs_model, spot_range, vol_range, strike, call_purchase_price, put_purchase_price)
    st.pyplot(heatmap_fig_call)

with col2:
    st.subheader("Put Option P&L Heatmap")
    _, heatmap_fig_put = plot_pnl_heatmap(bs_model, spot_range, vol_range, strike, call_purchase_price, put_purchase_price)
    st.pyplot(heatmap_fig_put)
