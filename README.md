# Customer Segmentation using K-Means Clustering

This project focuses on performing customer segmentation using the K-Means clustering algorithm. The goal is to group customers based on their purchasing behavior, allowing businesses to tailor their marketing strategies effectively.

## Overview

Customer segmentation is a crucial strategy for businesses aiming to understand their customer base better. By segmenting customers into distinct groups based on their characteristics, businesses can design personalized marketing campaigns, improve customer satisfaction, and increase revenue.

In this project, we utilize the K-Means clustering algorithm to segment customers based on key features such as annual income, spending score, age, profession, and family size.

## Dataset

The dataset used for this project contains information about 2,000 customers, including the following features:

- `CustomerID`: Unique ID for each customer
- `Gender`: Gender of the customer
- `Age`: Age of the customer
- `Annual Income ($)`: Annual income of the customer
- `Spending Score (1-100)`: Score assigned based on customer behavior and spending nature
- `Profession`: The profession of the customer
- `Work Experience`: Number of years the customer has been working
- `Family Size`: The size of the customer’s family

### Source

The dataset is sourced from Kaggle: [Customer Segmentation Dataset](https://www.kaggle.com/datasets/armanmanteghi/customer-segmentation-clustering-algorithm?select=Customers.csv).

## Acknowledgements

This project idea and dataset were inspired by [Arman Manteghi](https://www.linkedin.com/in/arman-manteghi-477858163/). We would like to thank him for providing the dataset and inspiration for this project.

## Project Structure

The project consists of the following steps:

1. **Data Loading and Exploration**: 
    - Load the dataset and perform an initial exploration to understand the data structure.
    - Assess the dataset for any missing values or duplicated entries.

2. **Data Cleaning**:
    - Rename columns for clarity.
    - Handle missing values by removing rows with null entries.
    - Identify and remove outliers based on the Interquartile Range (IQR) method.
    - Remove any inconsistent data, such as customers with an age lower than their work experience.

3. **Exploratory Data Analysis (EDA)**:
    - Visualize the distribution of numerical features using histograms and boxplots.
    - Analyze the distribution of categorical features using count plots.

4. **Clustering with K-Means**:
    - Perform data preprocessing, including encoding categorical variables and scaling the features.
    - Use the Elbow Method to determine the optimal number of clusters.
    - Apply the K-Means algorithm to segment the customers into distinct groups.

5. **Visualization and Analysis**:
    - Visualize the clusters using scatter plots.
    - Summarize the characteristics of each cluster to understand customer segments.

## Installation

To run this project, you need to have Python installed along with the following libraries:
## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/RayhanLup1n/customer-segmentation.git
