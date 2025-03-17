import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os

df = pd.read_csv(r'/Users/Johnny/Desktop/crypto-portfolio-data-engineering-project/data/station2/stage_2_crypto_data.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.set_index(['date'])
exportpath = r'/Users/Johnny/Desktop/crypto-portfolio-data-engineering-project/data/station3'

# Function to plot the mean-variance efficient frontier
def Station_3_Model_Design_Basic(returns_data,exportpath, ra = 0.1 ):
    num_assets = len(returns_data.iloc[1,])
    # Calculate mean returns and covariance matrix
    returns_data = returns_data*100
    mean_returns = (returns_data).mean()
    cov_matrix = returns_data.cov()  

    # Define function to calculate portfolio performance
    def portfolio_performance(weights, mean_returns, cov_matrix):
        returns = np.dot(weights, mean_returns)
        std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return returns, std

    # Number of portfolios to simulate
    num_portfolios = 25000
    results = np.zeros((3, num_portfolios))
    weights_record = []

    # Run simulations
    num_assets = len(mean_returns)
    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        returns, std = portfolio_performance(weights, mean_returns, cov_matrix)
        results[0, i] = returns
        results[1, i] = std
        results[2, i] = results[0, i] / results[1, i]
        weights_record.append(weights)

    # Convert results array to DataFrame
    results_frame = pd.DataFrame(results.T, columns=['Return', 'Standard Deviation', 'Sharpe Ratio'])

    # Locate the portfolio with the highest Sharpe ratio
    max_sharpe_idx = results_frame['Sharpe Ratio'].idxmax()
    max_sharpe_portfolio = results_frame.loc[max_sharpe_idx]
    max_sharpe_weights = weights_record[max_sharpe_idx]

    # Locate the portfolio with the minimum standard deviation
    min_vol_idx = results_frame['Standard Deviation'].idxmin()
    min_vol_portfolio = results_frame.loc[min_vol_idx]
    min_vol_weights = weights_record[min_vol_idx]
    
    
    # Ooptim Functions #
    def mean_variance(weights):
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -(portfolio_return / portfolio_volatility)
    
    def minimum_variance(weights):
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return portfolio_volatility
    
    def mean_variance_risk_aversion(weights,ra=ra):
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_volatility = (np.dot(weights.T, np.dot(cov_matrix, weights)))
        return - (portfolio_return - ra/2 * portfolio_volatility)
    
    # Optional ##############
    def risk_parity(weights):
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        marginal_risk_contribution = np.dot(cov_matrix, weights) / portfolio_volatility
        risk_contributions = weights * marginal_risk_contribution
        risk_target = portfolio_volatility / num_assets
        return np.sum((risk_contributions - risk_target) ** 2)
    
    def max_diversification(weights):
        individual_volatilities = np.sqrt(np.diag(cov_matrix))
        weighted_volatility = np.dot(weights, individual_volatilities)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        diversification_ratio = weighted_volatility / portfolio_volatility
        return -diversification_ratio
    #########################
    
    # Constraints and bounds
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))
    
    # Initial guess (equal weights)
    initial_guess = num_assets * [1. / num_assets,]
    
    # Optimize for MVO
    result = minimize(mean_variance, initial_guess, method='SLSQP', 
                      bounds=bounds, constraints=constraints)
    mvo_weights = np.round(result.x,3)
    
    rp_mvo, sigma_mvo = portfolio_performance(mvo_weights, mean_returns, cov_matrix)
    sr_mvo            =  rp_mvo/sigma_mvo
    
    # Optimize for MV
    result = minimize(minimum_variance, initial_guess, method='SLSQP', 
                      bounds=bounds, constraints=constraints)
    mv_weights =np.round( result.x , 3)
    
    rp_mv, sigma_mv = portfolio_performance(mv_weights, mean_returns, cov_matrix)
    sr_mv           = rp_mv/sigma_mv
    
    # Optimize for MVO 
    result = minimize(mean_variance_risk_aversion, initial_guess, method='SLSQP', 
                      bounds=bounds, constraints=constraints)
    mvrp_weights =np.round( result.x , 3)
    
    rp_mvrp, sigma_mvrp = portfolio_performance(mvrp_weights, mean_returns, cov_matrix)
    sr_mvrp             = rp_mvrp/sigma_mvrp
    
    # 1/N portfolio 
    ew_weights = np.array( [1/num_assets] * num_assets )
    rp_ew, sigma_ew     = portfolio_performance(ew_weights, mean_returns, cov_matrix)
    sr_ew               = rp_ew/sigma_ew
        
    # Plot the efficient frontier
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(results_frame['Standard Deviation'], results_frame['Return'], c=results_frame['Sharpe Ratio'], cmap='viridis')
    cbar = plt.colorbar(scatter)
    cbar.set_label('Sharpe Ratio', fontsize=14)
    cbar.ax.tick_params(labelsize=14)  # Increase the font size of the color bar ticks
    plt.scatter(sigma_mvo , rp_mvo  , marker='*', color='#FFA500', s=200, label='Max Sharpe Ratio')
    plt.scatter(sigma_mv  , rp_mv   , marker='*', color='b', s=200, label='Min Volatility')
    plt.scatter(sigma_mvrp, rp_mvrp , marker='*', color='g', s=200, label='Mean Variance RA = 0.1')
    plt.scatter(sigma_ew  , rp_ew   , marker='x', color='r', s=200, label='Equal Weights')
    plt.title('Mean-Variance Efficient Frontier', fontsize=14)
    plt.xlabel('Standard Deviation', fontsize=14)
    plt.ylabel('Return', fontsize=14)
    plt.legend(loc='upper left', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    # Save the figure
    filename = 'mean_variance_efficient_frontier.png'
    plt.savefig(os.path.join(exportpath, filename))   
    plt.show()
    
    # Output weights for the different portfolios #
    weights = pd.DataFrame({
        'MSR Weights':  mvo_weights,
        'MV Weights':   mv_weights,
        'MVRA Weights': mvrp_weights,
        'EW Weights'  : ew_weights
    }, index = df.columns)
    
    weights.index.rename('Stocks',inplace=True)
    weights = weights.reset_index()
    
    # Plot optimal weights #
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.2  # Adjust bar width to fit four bars comfortably
    index = range(len(weights['Stocks']))

    # Create bars for each set of weights
    bar1 = plt.bar(index, weights['MSR Weights'], bar_width, label='MSR Weights')
    bar2 = plt.bar([i + bar_width for i in index], weights['MV Weights'], bar_width, label='MV Weights')
    bar3 = plt.bar([i + bar_width * 2 for i in index], weights['MVRA Weights'], bar_width, label='MVRA Weights')
    bar4 = plt.bar([i + bar_width * 3 for i in index], weights['EW Weights'], bar_width, label='EW Weights')

    # Adding titles and labels
    plt.xlabel('Stocks', fontsize=14)
    plt.ylabel('Weights', fontsize=14)
    plt.title('Optimal Portfolio Weights Comparison', fontsize=15)
    plt.xticks([i + bar_width * 1.5 for i in index], weights['Stocks'], fontsize=14)
    plt.yticks(fontsize=14)

    # Adding a legend
    plt.legend(loc='upper left', fontsize=14)

    # Save the figure
    filename = 'optimal_weights.png'
    plt.savefig(os.path.join(exportpath, filename))   
    plt.show()
    
    # Compute optimal portfolio returns #
    weights = weights.set_index('Stocks')
    portfolio_returns = (returns_data @ weights)/100 # decimal format

    # Rename columns to reflect the portfolio strategies
    portfolio_returns.columns = ['MSR Portfolio', 'MV Portfolio', 'MVRA Portfolio','EW Portfolio']

    # Calculate cumulative returns
    cumulative_returns = (1 + portfolio_returns).cumprod() - 1
    
    # Plot cumulative returns over time
    plt.figure(figsize=(10, 8))
    for column in cumulative_returns.columns:
        plt.plot(cumulative_returns.index, cumulative_returns[column], label=column)
    
    # Customize the plot
    plt.title('Cumulative Optimal Portfolio Returns Over Time', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Cumulative Return', fontsize=14)
    plt.legend(title='Stock', fontsize=14, title_fontsize=14)
    plt.xticks(rotation=45, fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    
    # Save the figure
    filename = 'cumulative_optimal_portfolio_return.png'
    plt.savefig(os.path.join(exportpath, filename))   
    plt.show()
    
    # Descriptive statistics of the portfolios #
    descriptive_stats = (portfolio_returns*100).describe().transpose().round(3)
    descriptive_stats['Sharpe'] = (descriptive_stats['mean']/descriptive_stats['std']).round(3)
    
    cols = list(descriptive_stats.columns)
    sharpe_col = cols.pop(cols.index('Sharpe'))
    cols.insert(3, sharpe_col)
    descriptive_stats =  descriptive_stats[cols]
    
    # Export descriptive statistics #
    descriptive_stats.to_csv(os.path.join(exportpath, 'optimal_descriptive_statistics.csv'))
    # Export Weights #
    weights.to_csv(os.path.join(exportpath, 'optimal_weights.csv'))
      
    return list([weights, portfolio_returns, descriptive_stats])
        
##################### Execute the function #####################
Station3_output = Station_3_Model_Design_Basic(df,exportpath, ra = 0.1 )
###########################################################
########################### END ###########################
