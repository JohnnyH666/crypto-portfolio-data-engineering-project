import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import shapiro
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import os
from scipy.cluster.hierarchy import dendrogram, linkage
import nltk
from collections import Counter
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report
from tqdm import tqdm
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
nltk.download('punkt_tab')

# Inputs and export #
filepath   = r'/Users/Johnny/Desktop/crypto-portfolio-data-engineering-project/data/station1/stage_1_crypto_data.csv'
exportpath = r'/Users/Johnny/Desktop/crypto-portfolio-data-engineering-project/data/station2'

def station2_feature_engineering(filepath, exportpath, gen_plots=False):
    # Step 1: Read the data
    df = pd.read_csv(filepath)
    
    # 0 NaN checker #
    df.isnull().sum()
    df['close'] = np.where(df['close'] == 0,np.nan, df['close'])
    
    # Step 2: Pivot the data
    df = df.pivot_table(index = 'date',columns = 'ticker', values = 'close').reset_index()

    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index(['date'])
    
    # Drop columns if it has a NaN #
    df = df.dropna(axis=1)
   
    # Step 3: Compute returns
    df = df.pct_change().dropna()
    
    if gen_plots == True:

        # Check for missing values
        missing_values = df.isnull().sum()
    
        # Check data types
        data_types = df.dtypes
    
        # Step 4: Compute descriptive statistics
        # Calculate descriptive statistics
        # Multiple by 100 to get in percentage (daily)
        descriptive_stats = (df*100).describe().transpose().round(3)
    
        # Add skewness and kurtosis
        descriptive_stats['skewness'] = df.skew()
        descriptive_stats['kurtosis'] = df.kurtosis()
    
        # Conduct Shapiro-Wilk test for normality
        normality_tests = {}
        for crypto in df.columns:  # Skip the date column
            stat, p_value = shapiro(df[crypto])
            normality_tests[crypto] = {'Shapiro-Wilk Statistic': np.round(stat,3), 'p-value': np.round(p_value,3)}
            
        normality_tests = pd.DataFrame(normality_tests)

        '''
        # Plot for Outliers #
        '''
        # Function to identify outliers using the IQR method
        def identify_outliers(df, column):
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        
        # Identify outliers for each crypto and store the results
        outliers = {crypto: identify_outliers(df, crypto) for crypto in df.columns}
    
        # Generate box plots for visual inspection of outliers
        plt.figure(figsize=(16, 20))
        
        for i, crypto in enumerate(df.columns, 1):
            plt.subplot(len(df.columns) // 2 + 1, 2, i)
            sns.boxplot(y=df[crypto]*100)
            plt.title(f'Box plot for {crypto}', fontsize=18)
            plt.ylabel('Daily Return (%)', fontsize=16)
            plt.yticks(fontsize=18)
        
        plt.suptitle('Box Plots for Visual Inspection of Outliers', fontsize=22)
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        # Save the figure
        filename = 'outliers.png'
        plt.savefig(os.path.join(exportpath, filename))
        plt.show()
    
        # Summarize the outliers
        outliers_summary = {crypto: len(outliers[crypto]) for crypto in outliers}
        
        outliers_summary_df = pd.DataFrame.from_dict(outliers_summary, orient='index', columns=['Number of Outliers'])

        """
        # Correlation Analyses #
        """  
        correlation_matrix = df.corr()   
        # Plot the heatmap of the correlation matrix
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, cbar_kws={'label': 'Correlation'})  
        # Increase font size of x and y ticks
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        # Increase font size of x and y labels
        plt.xlabel('')
        plt.ylabel('')   
        # Title
        plt.title('Correlation Matrix Heatmap', fontsize=18)  
        # Save the figure
        filename = 'correlation.png'
        plt.savefig(os.path.join(exportpath, filename))   
        plt.show()

        """
        # Clustering 
        """            
        # Compute the distance matrix using 1 - correlation
        distance_matrix = 1 - correlation_matrix
        
        # Perform hierarchical clustering using the linkage method
        Z = linkage(distance_matrix, 'ward')
        
        # Plot the dendrogram
        plt.figure(figsize=(12, 8))
        dendrogram(Z, labels=correlation_matrix.columns, leaf_rotation=90, leaf_font_size=14)
        plt.title('Dendrogram : Clustering using Correlation', fontsize=18)
        plt.xlabel('Crypto', fontsize=16)
        plt.ylabel('Distance', fontsize=16)
        
        # Increase font size of x and y ticks
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        
        # Save the figure
        filename = 'dendogram.png'
        plt.savefig(os.path.join(exportpath, filename))   
        plt.show()
        
        """
        # Time-series statistics #
        """
        plt.figure(figsize=(16, 20))
        for i, crypto in enumerate(df.columns, 1):
            plt.subplot(len(df.columns) // 2 + 1, 2, i)
            plt.plot(df.index, df[crypto])
            plt.title(f'Time Series for {crypto}', fontsize=14)
            plt.xlabel('Date', fontsize=14)
            plt.ylabel('Return', fontsize=14)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
        plt.tight_layout()
        # Save the figure
        filename = 'time_series.png'
        plt.savefig(os.path.join(exportpath, filename))   
        plt.show()
        
        # Calculate rolling mean and standard deviation (window of 12 months)
        rolling_stats = {}
        for crypto in df.columns:
            rolling_stats[crypto] = df[crypto].rolling(window=21).agg(['mean', 'std'])
        
        # Plot rolling mean and standard deviation for each crypto
        plt.figure(figsize=(16, 20))
        for i, crypto in enumerate(df.columns, 1):
            plt.subplot(len(df.columns) // 2 + 1, 2, i)
            plt.plot(df.index, rolling_stats[crypto]['mean'], label='Rolling Mean')
            plt.plot(df.index, rolling_stats[crypto]['std'], label='Rolling Std')
            plt.title(f'Rolling Statistics for {crypto}', fontsize=14)
            plt.xlabel('Date', fontsize=14)
            plt.ylabel('Return / Volatility', fontsize=14)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.legend()
        plt.tight_layout()
        # Save the figure
        filename = 'time_series_rolling.png'
        plt.savefig(os.path.join(exportpath, filename))   
        plt.show()


        # Recalculate the risk-adjusted metrics with date as the index
        # Assuming a risk-free rate of 0.02% per day
        risk_free_rate = 0.0000
        
        # Calculate the average daily return for each crypto
        average_returns = df.mean()
        
        # Calculate the standard deviation of daily returns for each crypto
        std_dev_returns = df.std()
        
        # Calculate the Sharpe Ratio for each crypto
        sharpe_ratios = (average_returns - risk_free_rate) / std_dev_returns
        
        # Calculate the Sortino Ratio for each crypto
        negative_returns = df.applymap(lambda x: min(0, x))
        std_dev_negative_returns = negative_returns.std()
        sortino_ratios = (average_returns - risk_free_rate) / std_dev_negative_returns
        
        # Calculate the Beta for each crypto
        market_return = df.mean(axis=1)
        betas = df.apply(lambda x: np.cov(x, market_return)[0, 1] / np.var(market_return), axis=0)
        
        # Compile results into a DataFrame
        risk_adjusted_metrics = pd.DataFrame({
            'Average Return (Annual)': np.round( average_returns*252*100,3 ),
            'Standard Deviation (Annual)': np.round(std_dev_returns*np.sqrt(252)*100,3),
            'Sharpe Ratio (Annual)': np.round(sharpe_ratios*np.sqrt(252) , 3),
            'Sortino Ratio (Annual)': np.round(sortino_ratios*np.sqrt(252),3),
            'Beta': betas
        })
        
        # Set up the plotting environment
        sns.set_theme(style="whitegrid")
        
        # Create a DataFrame for the Sharpe and Sortino Ratios
        ratio_data = risk_adjusted_metrics[['Sharpe Ratio (Annual)', 'Sortino Ratio (Annual)']].reset_index()
        ratio_data = ratio_data.melt(id_vars='ticker', var_name='Ratio Type', value_name='Value')
        
        # Plotting the bar plot
        plt.figure(figsize=(12, 6))
        barplot = sns.barplot(x='ticker', y='Value', hue='Ratio Type', data=ratio_data, palette='muted')
        
        # Customize the plot
        plt.title('Sharpe Ratio and Sortino Ratio for Each Crypto')
        plt.xlabel('Crypto')
        plt.ylabel('Ratio Value')
        plt.legend(title='Ratio Type')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the figure
        filename = 'sharpe_ratios.png'
        plt.savefig(os.path.join(exportpath, filename))   
        plt.show()
        
        # Plotting the bar plot for Beta
        plt.figure(figsize=(12, 6))
        beta_plot = sns.barplot(x=betas.index, y=betas.values, palette='muted')
        
        # Customize the plot
        plt.title('Beta for Each Crypto')
        plt.xlabel('Crypto')
        plt.ylabel('Beta Value')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the figure
        filename = 'betas.png'
        plt.savefig(os.path.join(exportpath, filename))   
        plt.show()
        
        # Calculate cumulative returns
        cumulative_returns = (1 + df).cumprod() - 1
        
        # Plot cumulative returns over time
        plt.figure(figsize=(10, 8))
        for column in cumulative_returns.columns:
            plt.plot(cumulative_returns.index, cumulative_returns[column], label=column)
        
        # Customize the plot
        plt.title('Cumulative Returns Over Time', fontsize=16)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Cumulative Return', fontsize=14)
        plt.legend(title='Crypto', fontsize=14, title_fontsize=14)
        plt.xticks(rotation=45, fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        
        # Save the figure
        filename = 'cumulative_return.png'
        plt.savefig(os.path.join(exportpath, filename))   
        plt.show()
        
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    df.to_csv( os.path.join(exportpath,'stage_2_crypto_data.csv')   )
    descriptive_stats.to_csv(  os.path.join(exportpath,'stage_2_descriptive_stats.csv')   )
    
    print("______________________________________")
    print(f"Data successfully saved to {filepath}")
    print("--------------------------------------")
        
    return df, descriptive_stats

def station2_feature_engineering_news(exportpath):
    # =============================================================================
    # 1: Read in (load) the merged data
    # =============================================================================
    # Load the dataset
    df = pd.read_csv(r'/Users/Johnny/Desktop/crypto-portfolio-data-engineering-project/data/station1/stage1_sentiment.csv')
    df['datem'] = pd.to_datetime(df['date'])

    # =============================================================================
    # 2: Feature engineering
    # =============================================================================

    # Select necessary columns for analysis
    select_cols = [
        'datem',
        'date',
        'id',
        'title',
        'body',
        'categories'
    ]
    df = df[select_cols]

    # Define a function to preprocess text
    def preprocess_text(text):
        if not isinstance(text, str):
            return ""
        
        # Remove URLs, mentions, and hashtags
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\@\w+|\#', '', text)
        
        # Tokenize the text
        tokens = word_tokenize(text)
        
        # Remove stop words
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
        
        # Remove special characters and punctuation
        filtered_tokens = [re.sub(r'[^a-zA-Z0-9]+', '', token) for token in filtered_tokens]
        
        # Remove empty tokens
        filtered_tokens = [token for token in filtered_tokens if token]
        
        # Lemmatize the tokens
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
        
        # Join the tokens back into a string
        processed_text = ' '.join(lemmatized_tokens)
        return processed_text

    # Convert columns to string type and fill NaNs with an empty string
    df['title'] = df['title'].astype(str).fillna('')
    df['body'] = df['body'].astype(str).fillna('')

    # Apply the preprocessing function to the 'title' and 'body' columns
    tqdm.pandas()  # Enable progress_apply
    df['title_clean'] = df['title'].progress_apply(preprocess_text)
    df['body_clean'] = df['body'].progress_apply(preprocess_text)

    # =============================================================================
    # 3: Export to file
    # =============================================================================
    # Define file name
    file_name = 'processed_data.csv'

    # Save the processed data to a CSV file
    df.to_csv(os.path.join(exportpath, file_name), index=False)
    return


#############################Execute the function#######################################
#df, descriptive_stats = station2_feature_engineering(filepath, exportpath, gen_plots=True)
station2_feature_engineering_news(exportpath)
##########################################################################################
########################### END ##########################################################