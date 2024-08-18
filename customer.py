# Load Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load The Dataset
df = pd.read_csv('Customers.csv')

# Assessing Raw Dataset
df.info()
print(f"Shape of the data: {df.shape}")
print("Number of Null Values: ")
print(df.isna().sum())
print(f"Number of Duplicated Values: {df.duplicated().sum()}")

# Describe Raw Data
print(df.describe())
print(df.describe(include='object'))

# Make a copy of the raw data for further processing
cleaned_df = df.copy()

# Cleaning Data on the copied dataset
# 1. Rename Columns
cleaned_df = cleaned_df.rename(columns={
    'CustomerID': 'Id',
    'Annual Income ($)': 'Income',
    'Spending Score (1-100)': 'SpendingScore',
    'Work Experience': 'WorkExperience',
    'Family Size': 'Family'
})

# 2. Drop Null Values
cleaned_df = cleaned_df.dropna()

# 3. Handling Outliers
numerical_columns = cleaned_df.select_dtypes(include='number').columns
Q1 = cleaned_df[numerical_columns].quantile(0.25)
Q3 = cleaned_df[numerical_columns].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = (cleaned_df[numerical_columns] < lower_bound) | (cleaned_df[numerical_columns] > upper_bound)
cleaned_df = cleaned_df[~outliers.any(axis=1)]

# 4. Removing Inconsistent Data
cleaned_df = cleaned_df[~(cleaned_df['Age'] <= cleaned_df['WorkExperience'])]

# Exploratory Data Analysis (EDA)
# 1. Visualize Numerical Columns
for col in numerical_columns:
    plt.figure(figsize=(12, 6))
    sns.histplot(cleaned_df[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.boxplot(x=cleaned_df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()

# 2. Visualize Categorical Columns
object_columns = cleaned_df.select_dtypes(include='object').columns
for col in object_columns:
    plt.figure(figsize=(12, 6))
    sns.countplot(x=cleaned_df[col])
    plt.title(f'Distribution of {col}')
    plt.xticks(rotation=45)
    plt.show()

# Save Cleaned Dataframe
cleaned_df.to_csv('New_Customers.csv', index=False)

# Clustering Model
# 1. Data Preprocessing
encoder = LabelEncoder()
for col in object_columns:
    cleaned_df[col] = encoder.fit_transform(cleaned_df[col])

scaler = StandardScaler()
scaled_df = scaler.fit_transform(cleaned_df.drop(columns=['Id']))

# 2. Heatmap Correlation
plt.figure(figsize=(10,6))
sns.heatmap(pd.DataFrame(scaled_df, columns=cleaned_df.columns[1:]).corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Heatmap Correlation')
plt.show()

# 3. Clustering with K-Means
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(scaled_df)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Applying K-Means
optimal_clusters = 4
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=42)
cleaned_df['Cluster'] = kmeans.fit_predict(scaled_df)

# Visualize the Clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=cleaned_df, x='Income', y='SpendingScore', hue='Cluster', palette='viridis', s=100)
plt.title('Customer Segmentation')
plt.show()

# Model Summary
# Summary Statistics for Each Cluster
cluster_summary = cleaned_df.groupby('Cluster').mean()
print(cluster_summary)

plt.figure(figsize=(10, 6))
sns.boxplot(x='Cluster', y='Income', data=cleaned_df)
plt.title('Income Distribution by Cluster')
plt.show()

# Customer Segmentation Project
# This script performs customer segmentation using K-Means clustering.

# Acknowledgements:
# The idea and dataset for this project were inspired by Arman Manteghi.
# Dataset: https://www.kaggle.com/datasets/armanmanteghi/customer-segmentation-clustering-algorithm?select=Customers.csv
# LinkedIn: https://www.linkedin.com/in/arman-manteghi-477858163/