import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Function to generate portfolios and simulate the efficient frontier
def generate_portfolios(mean_returns, cov_matrix, num_portfolios=10000, risk_free_rate=0):
    num_assets = len(mean_returns)
    results = np.zeros((4, num_portfolios))  # Added another row for Sharpe Ratio
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_std_dev, portfolio_return = portfolio_performance(weights, mean_returns, cov_matrix)
        results[0, i] = portfolio_std_dev
        results[1, i] = portfolio_return
        results[2, i] = (portfolio_return - risk_free_rate) / portfolio_std_dev  # Sharpe Ratio
    return results, weights_record

# Function to find the portfolio with the highest Sharpe Ratio
def max_sharpe_ratio_portfolio(results, weights_record):
    max_sharpe_idx = np.argmax(results[2])
    sdp, rp = results[0, max_sharpe_idx], results[1, max_sharpe_idx]
    max_sharpe_allocation = weights_record[max_sharpe_idx]
    return sdp, rp, max_sharpe_allocation

# Function to find the portfolio with the minimum volatility
def min_volatility_portfolio(results, weights_record):
    min_vol_idx = np.argmin(results[0])
    sdp, rp = results[0, min_vol_idx], results[1, min_vol_idx]
    min_vol_allocation = weights_record[min_vol_idx]
    return sdp, rp, min_vol_allocation

# Helper function to calculate portfolio performance
def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns * weights) * 252
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return volatility, returns

def calculate_log_returns(prices):
    """
    Calculate the log returns for a DataFrame of asset prices.

    Parameters:
    - prices: DataFrame containing the prices of assets with assets in columns and dates in rows.

    Returns:
    - DataFrame containing log returns of the assets.
    """
    log_returns = np.log(prices / prices.shift(1))
    return log_returns.dropna()  # Optionally, remove the first row which will be NaN


# PLOTS

import matplotlib.pyplot as plt

def plot_efficient_frontier(results):
    plt.scatter(results[0], results[1], c=results[2], cmap='viridis')
    plt.colorbar(label='Sharpe Ratio')
    plt.xlabel('Volatility')
    plt.ylabel('Return')
    plt.title('Efficient Frontier')
    plt.show()

def plot_log_returns(log_returns):
    plt.figure(figsize=(14, 7))
    for c in log_returns.columns.values:
        plt.plot(log_returns.index, log_returns[c], label=c)
    plt.title('Asset Log Returns')
    plt.xlabel('Date')
    plt.ylabel('Log return')
    plt.legend(loc='best')
    plt.show()

def plot_efficient_frontier_with_highlights(results, max_sharpe_allocation, min_vol_allocation):
    plt.figure(figsize=(10, 7))
    plt.scatter(results[0], results[1], c=results[2], cmap='viridis', label='Efficient Frontier')
    plt.colorbar(label='Sharpe Ratio')
    plt.xlabel('Volatility')
    plt.ylabel('Return')
    plt.title('Efficient Frontier with Highlighted Portfolios')
    
    # Highlight the maximum Sharpe ratio portfolio
    max_sharpe_volatility = max_sharpe_allocation[0]
    max_sharpe_return = max_sharpe_allocation[1]
    plt.scatter(max_sharpe_volatility, max_sharpe_return, color='red', marker='*', s=500, label='Maximum Sharpe Ratio Portfolio')
    
    # Highlight the minimum volatility portfolio
    min_vol_volatility = min_vol_allocation[0]
    min_vol_return = min_vol_allocation[1]
    plt.scatter(min_vol_volatility, min_vol_return, color='blue', marker='*', s=500, label='Minimum Volatility Portfolio')
    
    plt.legend(labelspacing=0.8)
    plt.show()


def plot_asset_performance(log_returns):
    """
    Plots the annualized return and annualized volatility for each asset in the log_returns DataFrame.

    Parameters:
    - log_returns: DataFrame containing log returns of assets.
    """
    # Calculate annualized return and annualized volatility
    annualized_return = log_returns.mean() * 252
    annualized_volatility = log_returns.std() * np.sqrt(252)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    for i, txt in enumerate(annualized_return.index):
        plt.scatter(annualized_volatility[i], annualized_return[i], label=txt)
        plt.text(annualized_volatility[i], annualized_return[i], txt, fontsize=9)
    
    plt.title('Asset Performance: Return vs. Volatility')
    plt.xlabel('Annualized Volatility (Risk)')
    plt.ylabel('Annualized Return')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()
    
def plot_portfolio_allocation(weights, title):
    # Ensure weights are normalized (sum to 1) if not already
    weights = [float(i)/sum(weights) for i in weights]
    
    plt.figure(figsize=(10, 7))
    plt.pie(weights, labels=asset_names, autopct='%1.1f%%', startangle=140)
    plt.title(title)
    plt.show()


import pandas as pd


# Load the data
bond_eur = pd.read_csv('./Data/BOND_EUR Kraken Historical Data.csv', index_col='Date', parse_dates=True)
crude_oil_wti = pd.read_csv('./Data/Crude Oil WTI Futures - Apr 24 (CLJ4).csv', index_col='Date', parse_dates=True)
india_bond = pd.read_csv('./Data/India 3-Month Bond Yield Historical Data.csv', index_col='Date', parse_dates=True)
namibia_bond = pd.read_csv('./Data/Namibia 3-Month Bond Yield Historical Data.csv', index_col='Date', parse_dates=True)
s_p_500 = pd.read_csv('./Data/S&P 500 (US500).csv', index_col='Date', parse_dates=True)
sbi_gold_etf = pd.read_csv('./Data/SBI Gold ETF (SBIG).csv', index_col='Date', parse_dates=True)
spdr_dow_jones = pd.read_csv('./Data/SPDR® Dow Jones Industrial Average ETF Trust (SPDR).csv', index_col='Date', parse_dates=True)
uk_10yr_gilt = pd.read_csv('./Data/UK 10 YR Gilt Futures Historical Data.csv', index_col='Date', parse_dates=True)
us_soybeans = pd.read_csv('./Data/US Soybeans Futures - Mar 24 (ZSH4).csv', index_col='Date', parse_dates=True)
us_wheat = pd.read_csv('./Data/US Wheat Futures - Mar 24 (ZWH4).csv', index_col='Date', parse_dates=True)

# Forward fill the missing values for each asset before combining them
bond_eur.ffill(inplace=True)
crude_oil_wti.ffill(inplace=True)
india_bond.ffill(inplace=True)
namibia_bond.ffill(inplace=True)
s_p_500.ffill(inplace=True)
sbi_gold_etf.ffill(inplace=True)
spdr_dow_jones.ffill(inplace=True)
uk_10yr_gilt.ffill(inplace=True)
us_soybeans.ffill(inplace=True)
us_wheat.ffill(inplace=True)

# Combine into a single DataFrame
prices = pd.concat([
    bond_eur['Price'], 
    crude_oil_wti['Price'], 
    india_bond['Price'], 
    namibia_bond['Price'], 
    s_p_500['Price'], 
    sbi_gold_etf['Price'], 
    spdr_dow_jones['Price'], 
    uk_10yr_gilt['Price'], 
    us_soybeans['Price'], 
    us_wheat['Price']
], axis=1)

# Rename columns
prices.columns = [
    'BOND_EUR', 
    'CRUDE_OIL_WTI', 
    'INDIA_BOND', 
    'NAMIBIA_BOND', 
    'S_P_500', 
    'SBI_GOLD_ETF', 
    'SPDR_DOW_JONES', 
    'UK_10YR_GILT', 
    'US_SOYBEANS', 
    'US_WHEAT'
]


prices = prices.apply(pd.to_numeric, errors='coerce')
# Apply interpolation to fill in any remaining gaps
prices.interpolate(method='linear', inplace=True)
# Check for non-overlapping dates
print(prices.count())  # This will show you the count of non-NaN values per column



import matplotlib.pyplot as plt

# Plotting the log returns
log_returns = np.log(prices / prices.shift(1)).dropna()
print("Number of rows in log_returns:", len(log_returns))

# Proceed with further analysis such as plotting log returns, calculating mean returns and volatility, etc.

mean_returns = log_returns.mean()
cov_matrix = log_returns.cov()
plot_log_returns(log_returns)


np.random.seed(76) 
results, weights_record = generate_portfolios(mean_returns, cov_matrix)
# Portfolio with the Highest Sharpe Ratio
sdp_max_sharpe, rp_max_sharpe, max_sharpe_allocation = max_sharpe_ratio_portfolio(results, weights_record)

# Portfolio with the Minimum Volatility
sdp_min_vol, rp_min_vol, min_vol_allocation = min_volatility_portfolio(results, weights_record)
# Adjust the parameters to include volatility and return values for the portfolios
max_sharpe_allocation = [sdp_max_sharpe, rp_max_sharpe]
min_vol_allocation = [sdp_min_vol, rp_min_vol]

plot_efficient_frontier_with_highlights(results, max_sharpe_allocation, min_vol_allocation)

# Example usage with a hypothetical 'log_returns' DataFrame
plot_asset_performance(log_returns)

import matplotlib.pyplot as plt

# Assuming asset_names matches the order of assets in your weights_record
asset_names = prices.columns


# Plotting the pie charts for the optimal portfolios
plot_portfolio_allocation(max_sharpe_ratio_portfolio(results, weights_record)[2], 'Portfolio with Maximum Sharpe Ratio')

plot_portfolio_allocation(min_volatility_portfolio(results, weights_record)[2], 'Portfolio with Minimum Volatility')


market_log_returns = log_returns['S_P_500']
betas = {}

for asset in log_returns.columns:
    if asset != 'S_P_500':  # Exclude the market itself
        covariance = log_returns[asset].cov(market_log_returns)
        market_variance = market_log_returns.var()
        betas[asset] = covariance / market_variance

        
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Assuming s_p_500 is already loaded with 'Price' as one of the columns and indexed by 'Date'
# Recalculating the log returns for S&P 500 for clarity
s_p_500['Log_Returns'] = np.log(s_p_500['Price'] / s_p_500['Price'].shift(1))

# Plotting S&P 500 Price and Log Returns on separate subplots
fig, ax1 = plt.subplots(figsize=(14, 7))

color = 'tab:red'
ax1.set_xlabel('Date')
ax1.set_ylabel('S&P 500 Price', color=color)
ax1.plot(s_p_500.index, s_p_500['Price'], color=color, label='S&P 500 Price')
ax1.tick_params(axis='y', labelcolor=color)

# Instantiate a second axes that shares the same x-axis
ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('Log Returns', color=color)  
ax2.plot(s_p_500.index, s_p_500['Log_Returns'], color=color, label='S&P 500 Log Returns')
ax2.tick_params(axis='y', labelcolor=color)

# Adding a title and customizing layout
plt.title('S&P 500 Price and Log Returns Over Time')
fig.tight_layout()  

plt.show()


R_f = 0.010
R_m = market_log_returns.mean() * 252  # Annualizing the average daily log return of the market

expected_returns = {}
for asset, beta in betas.items():
    expected_returns[asset] = R_f + beta * (R_m - R_f)

    
for asset, exp_return in expected_returns.items():
    print(f"{asset}: Expected Annual Return = {exp_return:.2%}")

    
