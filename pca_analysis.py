#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PCA Analysis for SPY Data

This script loads the full SPYV3.csv dataset, drops unnecessary columns, factorizes categorical variables,
applies two types of scaling (MinMax and StandardScaler), and performs Principal Component Analysis (PCA)
using the MinMax scaled data (switch to StandardScaler data if desired). It prints the explained variance ratio,
the cumulative variance for the selected number of components, and displays the PCA components (loadings).
Finally, it applies filters to select features with high contributions in each principal component.

Author: Pablo Beret
Created on Sat Sep 15 14:03:18 2018 (Updated version)
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --------------------------
# Data Loading and Preprocessing
# --------------------------
# Load the full dataset; drop columns that are not needed
data = pd.read_csv("SPYV3.csv", sep=',')
columns_to_drop = ['FECHA','OPEN', 'MAX', 'MIN', 'CLOSE','CLASIFICADOR', 
                   'FECHA.year', 'FECHA.day-of-month', 'FECHA.day-of-week']
data = data.drop(columns=columns_to_drop)

# Factorize categorical variables that are not numeric
categorical_vars = ['39', '41', '43', '168', '172']
for var in categorical_vars:
    data[var], _ = pd.factorize(data[var])

# --------------------------
# Scaling the Data
# --------------------------
# Scale data using MinMaxScaler
min_max_scaler = preprocessing.MinMaxScaler()
data_minmax = min_max_scaler.fit_transform(data)
data_minmax = pd.DataFrame(data_minmax, columns=data.columns)

# Scale data using StandardScaler (if needed)
data_standard = StandardScaler().fit_transform(data)

# --------------------------
# Principal Component Analysis (PCA)
# --------------------------
# For PCA, we use the MinMax scaled data. To use StandardScaler data, replace data_minmax with data_standard.
n_components = 4
pca_estimator = PCA(n_components=n_components)
X_pca = pca_estimator.fit_transform(data_minmax)

# Print the explained variance ratio of each component
print("Explained Variance Ratio per Component:")
print(pca_estimator.explained_variance_ratio_)

# Calculate cumulative explained variance
cumulative_variance = np.sum(pca_estimator.explained_variance_ratio_)
print("\nTotal Explained Variance for {} components: {:.4f}".format(n_components, cumulative_variance))

# Create a DataFrame for the PCA loadings (components)
pca_loadings = pd.DataFrame(np.transpose(pca_estimator.components_), 
                            columns=[f'PC-{i+1}' for i in range(n_components)], 
                            index=data.columns)
print("\nPCA Loadings:")
print(pca_loadings)

# --------------------------
# Filtering variables with high loadings in each principal component
# --------------------------
# (These thresholds can be adapted for each case)
filter_pc1 = pca_loadings[pca_loadings['PC-1'] >= 0.10]
print("\nVariables with PC-1 loading >= 0.10:")
print(filter_pc1)

filter_pc2 = pca_loadings[pca_loadings['PC-2'] >= 0.15]
print("\nVariables with PC-2 loading >= 0.15:")
print(filter_pc2)

filter_pc3 = pca_loadings[pca_loadings['PC-3'] >= 0.25]
print("\nVariables with PC-3 loading >= 0.25:")
print(filter_pc3)

filter_pc4 = pca_loadings[pca_loadings['PC-4'] >= 0.30]
print("\nVariables with PC-4 loading >= 0.30:")
print(filter_pc4)
