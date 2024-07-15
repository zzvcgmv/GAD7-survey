# GAD7-survey
import pandas as pd  
import numpy as np  
import ipinfo_db    
import plotly.express as px  
import matplotlib.pyplot as plt  
import matplotlib.cm as cm  
import seaborn as sns  
import scipy.stats as stats  
  
from math import pi  
from scipy.stats import chi2_contingency  
  
from mlxtend.frequent_patterns import apriori, association_rules  
from mlxtend.preprocessing import TransactionEncoder  
  
from sklearn.decomposition import PCA  
from sklearn.linear_model import LinearRegression  
from sklearn.metrics import mean_squared_error  
from sklearn.model_selection import train_test_split  
import statsmodels.api as sm  
  
from sklearn.preprocessing import StandardScaler  
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering  
from sklearn.mixture import GaussianMixture  
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score  
  
# Gepographic location converted from ip address  
class GeoLocator:    
    def __init__(self, access_token):    
        self.client = ipinfo_db.Client(access_token)    
    
    def get_geo_details(self, ip_addresses):    
        countries = []    
        provinces = []    
        for ip_address_string in ip_addresses:    
            ip_address = ip_address_string.split(',')[0].strip()  # Remove potential whitespace and extra data    
            details = self.client.getDetails(ip_address)    
            print(details)  # Optionally, you might want to remove or modify this in production    
    
            # Safely extracting 'country' and 'province' data    
            country = details.country_name if hasattr(details, 'country') else None    
            province = details.region if hasattr(details, 'region') else None    
                
            countries.append(country)    
            provinces.append(province)    
            
        return countries, provinces    
    
    def update_dataframe(self, df, ip_address_col):    
        countries, provinces = self.get_geo_details(df[ip_address_col])    
        df['Country'] = countries    
        df['Province'] = provinces    
        return df    
  
# Geographic Visualization  
class SurveyDataVisualizer:  
    def __init__(self, dataframe):  
        self.df = dataframe  
  
    def filter_by_country(self, country_name):  
        """ Filter the DataFrame for entries from a specific country. """  
        filtered_df = self.df[self.df['country'] == country_name]  
        return filtered_df  
  
    def province_distribution(self, country_name):  
        """ Generate a count of responses by province for a specified country. """  
        filtered_df = self.filter_by_country(country_name)  
        province_counts = filtered_df['province'].value_counts().reset_index()  
        province_counts.columns = ['province', 'count']  
        return province_counts  
  
    def plot_province_distribution(self, country_name):  
        """ Visualize the distribution of responses by province. """  
        province_counts = self.province_distribution(country_name)  
        fig = px.bar(province_counts, x='province', y='count', title=f'Number of Respondents by Province in {country_name}')  
        fig.update_layout(xaxis_title='Province', yaxis_title='Number of Respondents', showlegend=False)  
        fig.show()  
  
# Split the multiple-chocies into dummy variables  
  
class SurveyDataProcessor:  
    def __init__(self, dataframe):  
        self.df = dataframe  
  
    def split_and_encode(self, columns):  
        """ 
        Splits the string of multiple choice answers in the specified columns into separate dummy variables. 
        Args: 
        columns (list): A list of column names to process. 
        """  
        for col in columns:  
            if col in self.df.columns:  
                # Splits the string into a list, expands to columns, and converts NaN to False  
                dummy_df = self.df[col].astype(str).str.split('|', expand=True).stack().str.get_dummies().sum(level=0, axis=0)  
                # Converts numeric columns back to boolean  
                dummy_df = dummy_df.astype(bool)  
                # Drop the original column and join the new dummy variables  
                self.df = self.df.drop(columns=[col]).join(dummy_df.add_prefix(f'{col}_'))  
        return self.df  
  
    def preprocess(self):  
        split_columns = ['110', '500', '510', '520', '530', '540', '560', '580']  
        self.df = self.split_and_encode(split_columns)  
        return self.df  
  
# Bar plots for population descriptive visualization  
class barPlots(object):  
  def __init__(self, df, column, title, xlabel, ylabel):  
    self.df = df  
    self.column = column  
    self.title = title  
    self.xlabel = xlabel  
    self.ylabel = ylabel  
  
  def draw(self):  
    # Create a figure and an axes object  
    fig, ax = plt.subplots()  
  
    # Build a def for bar chart plots  
    def barPlots(df, column, title, xlabel, ylabel):  
      # Calculate categories counts  
      categories_counts = df[column].value_counts()  
  
      # Generate a colormap with a distinct color for each category  
      colors = cm.get_cmap('viridis', len(categories_counts))  
  
      # Plot the bar chart  
      ax = df[column].value_counts().plot(kind='bar', title=title, xlabel=xlabel, ylabel=ylabel, color=colors(range(len(categories_counts))))  
  
      def countLables(categories_counts): # Define categories_counts as parameter  
        # Add count labels to each bar  
        total_count = categories_counts.sum()  
        for i, count in enumerate(categories_counts):  
          proportion = count / total_count  
          label = f"{count} ({proportion:.4f})"  # Format proportion to four decimal places  
          ax.text(i, count, label, ha='center', va='bottom', fontsize='x-small')  
  
      countLables(categories_counts) # Pass categories_counts to countLables  
  
    barPlots(self.df, self.column, self.title, self.xlabel, self.ylabel)  
    plt.show()  
  
# Multiple-choice Visualization: Radar Plot  
class SurveyDataVisualizer:  
    def __init__(self, dataframe):  
        self.df = dataframe  
  
    def prepare_data_for_radar_chart(self, question_prefix):  
        """ Extracts and sums data for radar chart visualization based on question prefix. """  
        question_columns = [col for col in self.df.columns if question_prefix in col]  
        df_question = self.df[question_columns]  
        question_sums = df_question.sum()  
        return question_sums  
  
    def plot_radar_chart(self, question_prefix, labels, title):  
        """ Plots a radar chart for the specified question data. """  
        values = self.prepare_data_for_radar_chart(question_prefix)  
        num_vars = len(labels)  
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()  
  
        # Complete the loop  
        values = values.tolist() + values.tolist()[:1]  
        angles += angles[:1]  
  
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))  
        ax.fill(angles, values, color='r', alpha=0.25)  
        ax.plot(angles, values, color='r', linewidth=2)  
        ax.set_yticklabels([])  
        ax.set_xticks(angles[:-1])  
        ax.set_xticklabels(labels, fontsize=12)  
  
        plt.title(title, size=15)  
        plt.tight_layout()  
        plt.show()  
  
# Crosstabulation developing, heatmap visualization, and test statistics  
class SurveyDataAnalysis:  
    def __init__(self, dataframe):  
        self.df = dataframe  
  
    def generate_crosstab(self, question_x, question_y):  
        """Generate a crosstab from two specified columns."""  
        crosstab = pd.crosstab(self.df[question_x], self.df[question_y])  
        crosstab.index = ['Not at all', 'Several days', 'More than a week', 'Almost everyday']  
        crosstab.columns = ['Not at all', 'Several days', 'More than a week', 'Almost everyday']  
        return crosstab  
  
    def plot_heatmap(self, crosstab, title):  
        """Plot a heatmap from a crosstab."""  
        plt.figure(figsize=(10, 8))  
        sns.heatmap(crosstab, annot=True, fmt="d", cmap="viridis")  
        plt.title(title)  
        plt.xlabel('Categories')  
        plt.ylabel('Categories')  
        plt.tight_layout()  
        plt.show()  
  
    def chi_square_test(self, crosstab):  
        """Perform chi-square test and return results."""  
        chi2, p, dof, expected = chi2_contingency(crosstab)  
        return chi2, p, dof, expected  
  
    def symmetry_measurement(self, crosstab):  
        """Measure symmetry in a crosstab."""  
        diff = crosstab.values - crosstab.values.T  
        symmetry_score = np.sum(np.abs(diff)) / np.sum(crosstab.values)  
        return symmetry_score  
  
    def cramers_v(self, crosstab):  
        """Calculate Cramer's V for association between categorical variables."""  
        chi2, _, _, _ = chi2_contingency(crosstab)  
        n = crosstab.sum().sum()  
        return np.sqrt(chi2 / (n * (min(crosstab.shape) - 1)))  
  
# Univariate Association analysis with Apriori  
class AssociationAnalysis:  
    def __init__(self, dataframe):  
        self.df = dataframe  
  
    def prepare_data(self, columns):  
        """ Prepare data for Apriori by filtering columns and applying TransactionEncoder """  
        df_apriori = self.df[columns]  
        te = TransactionEncoder()  
        te_ary = te.fit(df_apriori).transform(df_apriori)  
        return pd.DataFrame(te_ary, columns=te.columns_)  
  
    def find_frequent_itemsets(self, dataframe, min_support=0.3):  
        """ Find frequent itemsets using the Apriori algorithm """  
        return apriori(dataframe, min_support=min_support, use_colnames=True)  
  
    def generate_rules(self, itemsets, metric="lift", min_threshold=1):  
        """ Generate association rules from frequent itemsets """  
        return association_rules(itemsets, metric=metric, min_threshold=min_threshold)  
  
    def perform_analysis(self, columns_prefix, supports, lift_thresholds):  
        """ Conducts complete analysis for different supports and lift thresholds """  
        # Filter columns based on prefix  
        columns = [col for col in self.df.columns if any(f'q{q}-' in col for q in columns_prefix)]  
        df_apriori = self.prepare_data(columns)  
  
        for support in supports:  
            frequent_itemsets = self.find_frequent_itemsets(df_apriori, min_support=support)  
            print(f"\nSupport: {support}")  
            if not frequent_itemsets.empty:  
                print("Frequent Itemsets:")  
                print(frequent_itemsets)  
                for threshold in lift_thresholds:  
                    rules = self.generate_rules(frequent_itemsets, min_threshold=threshold)  
                    print(f"Threshold: {threshold}")  
                    print("Association Rules:")  
                    print(rules)  
            else:  
                print("No frequent itemsets found for this support.")  
  
# Bivariate Association Analysis: Pearson  
class DataAnalyzer:  
    def __init__(self, dataframe):  
        self.df = dataframe  
  
    def calculate_pearson_correlation(self, columns):  
        """ Calculate Pearson correlation coefficients and p-values for given columns """  
        pearson_corr = self.df[columns].corr(method='pearson')  
        p_value = self.df[columns].corr(method=lambda x, y: stats.pearsonr(x, y)[1])  
  
        # Fill diagonal and lower triangle of the p_value matrix  
        p_value = pd.DataFrame(p_value, index=pearson_corr.index, columns=pearson_corr.columns)  
        for col in columns:  
            for row in columns:  
                if col != row:  
                    corr, p_val = stats.pearsonr(self.df[col], self.df[row])  
                    pearson_corr.loc[col, row] = corr  
                    p_value.loc[col, row] = p_val  
                else:  
                    pearson_corr.loc[col, row] = 1.0  
                    p_value.loc[col, row] = 0.0  
        return pearson_corr, p_value  
  
    def plot_correlation_heatmap(self, correlation_matrix, title, cmap='coolwarm', center=0):  
        """ Plot a heatmap for the correlation matrix """  
        plt.figure(figsize=(10, 8))  
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap=cmap, center=center)  
        plt.title(title)  
        plt.show()  
  
# Optional: Combine correlation coefficients and p-values into annotations  
annotations = pearson_corr.round(2).astype(str) + "\n(p=" + p_values.round(3).astype(str) + ")"  
analyzer.plot_correlation_heatmap(pearson_corr, 'Pearson Correlation Heatmap', annotations)  
  
# Contingency tables for 'Occupation' against other qualitative variables  
class ContingencyTableAnalyzer:  
    def __init__(self, dataframe):  
        self.df = dataframe  
  
    def create_contingency_tables(self, base_column, other_columns):  
        """Create contingency tables comparing a base column with other columns."""  
        contingency_tables = {}  
        for col in other_columns:  
            if col != base_column:  
                contingency_tables[col] = pd.crosstab(self.df[base_column], self.df[col])  
        return contingency_tables  
  
    def display_tables(self, tables):  
        """Display each contingency table."""  
        for col, table in tables.items():  
            print(f"Contingency Table for {col}:\n{table}\n")  
  
    def plot_heatmaps(self, tables):  
        """Visualize contingency tables using heatmaps."""  
        for col, table in tables.items():  
            plt.figure(figsize=(10, 8))  
            sns.heatmap(table, annot=True, fmt='d', cmap='YlGnBu')  
            plt.title(f'Contingency Table Heatmap for {col}')  
            plt.xlabel(col)  
            plt.ylabel('Occupation')  
            plt.show()  
  
# Regression Analysis  
class RegressionAnalysis:  
    def __init__(self, dataframe):  
        self.df = dataframe  
  
    def clean_data(self):  
        """Check and handle missing values."""  
        missing_values = self.df.isnull().sum()  
        print(f'Missing values in each column:\n{missing_values}\n')  
        self.df = self.df.dropna()  
        return self.df  
  
    def fit_regression_model(self, dependent_var, independent_vars):  
        """Fit a regression model and check VIF."""  
        X = self.df[independent_vars]  
        X = sm.add_constant(X)  # adding a constant  
        y = self.df[dependent_var]  
        model = sm.OLS(y, X).fit()  
        print(model.summary())  
        return model, X  
  
    def check_multicollinearity(self, X):  
        """Calculate VIF for each independent variable."""  
        vif_data = pd.DataFrame()  
        vif_data["feature"] = X.columns  
        vif_data["VIF"] = [sm.stats.outliers_influence.variance_inflation_factor(X.values, i) for i in range(X.shape[1])]  
        print(vif_data)  
  
    def plot_residuals(self, model):  
        """Plot residuals to check for homoscedasticity and normality."""  
        residuals = model.resid  
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))  
        sns.histplot(residuals, kde=True, ax=ax[0])  
        ax[0].set_title('Residuals Distribution')  
        sm.qqplot(residuals, line='s', ax=ax[1])  
        ax[1].set_title('Q-Q Plot')  
        plt.show()  
  
    def apply_pca_and_fit_model(self, independent_vars, dependent_var, n_components):  
        """Apply PCA to the data and fit a regression model on the components."""  
        pca = PCA(n_components=n_components)  
        X_pca = pca.fit_transform(self.df[independent_vars])  
        X_train, X_test, y_train, y_test = train_test_split(X_pca, self.df[dependent_var], test_size=0.3, random_state=42)  
          
        model_pca = LinearRegression()  
        model_pca.fit(X_train, y_train)  
        y_pred = model_pca.predict(X_test)  
        mse = mean_squared_error(y_test, y_pred)  
        rmse = np.sqrt(mse)  
        r_squared = model_pca.score(X_test, y_test)  
  
        print(f'Root Mean Squared Error: {rmse}')  
        print(f'R-squared: {r_squared}')  
        print(f'PCA Components:\n{pca.components_}')  
        print(f'Explained Variance Ratio: {pca.explained_variance_ratio_}')  
        return model_pca, X_train, y_train, X_test, y_test  
  
# PCA and KMeans clustering on demographic data  
class PopulationClustering:  
    def __init__(self, dataframe, demographic_columns):  
        self.df = dataframe  
        self.demographic_columns = demographic_columns  
        self.model = None  
        self.pca = None  
        self.clusters = None  
  
    def standardize_data(self):  
        """Standardize demographic data."""  
        scaler = StandardScaler()  
        return scaler.fit_transform(self.df[self.demographic_columns])  
  
    def perform_pca(self, standardized_data, n_components=2):  
        """Perform PCA to reduce dimensionality."""  
        self.pca = PCA(n_components=n_components)  
        return self.pca.fit_transform(standardized_data)  
  
    def cluster_data(self, pca_data, n_clusters=5):  
        """Perform KMeans clustering."""  
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)  
        self.clusters = kmeans.fit_predict(pca_data)  
        self.df['Cluster'] = self.clusters  
        return self.clusters  
  
    def plot_clusters(self, pca_data):  
        """Plot the PCA results with cluster coloring."""  
        plt.figure(figsize=(10, 6))  
        sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue=self.clusters, palette='viridis', s=100)  
        plt.title('PCA of Demographic Data with KMeans Clusters')  
        plt.xlabel('PCA Component 1')  
        plt.ylabel('PCA Component 2')  
        plt.legend(title='Cluster')  
        plt.show()  
  
    def analyze_clusters(self):  
        """Analyze and display cluster characteristics."""  
        cluster_analysis = self.df.groupby('Cluster')[self.demographic_columns].mean()  
        print("Cluster Analysis Results:\n", cluster_analysis)  
        plt.figure(figsize=(12, 8))  
        sns.heatmap(cluster_analysis, annot=True, cmap='coolwarm', fmt=".2f")  
        plt.title('Cluster Analysis of Demographic Variables')  
        plt.show()  
  
# Clustering Algorithms: KMeans, Hierarchical, DBSCAN, GMM  
class ClusterEvaluator:  
    def __init__(self, data, algorithms):  
        self.data = data  
        self.algorithms = algorithms  
        self.results = {}  
          
    def fit_and_evaluate(self):  
        for name, algo in self.algorithms.items():  
            # Fit model  
            if hasattr(algo, 'fit_predict'):  
                labels = algo.fit_predict(self.data)  
            else:  
                algo.fit(self.data)  
                labels = algo.labels_  
              
            # Evaluate model  
            silhouette = silhouette_score(self.data, labels)  
            davies_bouldin = davies_bouldin_score(self.data, labels)  
            calinski_harabasz = calinski_harabasz_score(self.data, labels)  
              
            # Store results  
            self.results[name] = {  
                'labels': labels,  
                'silhouette': silhouette,  
                'davies_bouldin': davies_bouldin,  
                'calinski_harabasz': calinski_harabasz  
            }  
      
    def display_results(self):  
        for name, metrics in self.results.items():  
            print(f"\n{name} Clustering Results:")  
            print(f"Silhouette Score: {metrics['silhouette']}")  
            print(f"Davies-Bouldin Score: {metrics['davies_bouldin']}")  
            print(f"Calinski-Harabasz Score: {metrics['calinski_harabasz']}")  
