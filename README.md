# PCA Analysis for SPY Data

This repository provides an example of Principal Component Analysis (PCA) in Python applied to SPY stock index data. The script demonstrates how to load and preprocess the dataset, perform scaling using MinMaxScaler (with an option for StandardScaler), and run PCA on the scaled data. Additionally, it shows how to extract and filter the PCA loadings (components) to identify features with high contributions.

## Features

- **Data Preprocessing:**  
  Loads the complete dataset and removes unnecessary columns.
  
- **Categorical Variable Factorization:**  
  Factorizes categorical variables that are non-numeric.
  
- **Data Scaling:**  
  Applies MinMax scaling (and provides an option for Standard scaling).
  
- **Principal Component Analysis:**  
  Performs PCA on the scaled data, prints the explained variance ratio, and computes the cumulative explained variance.
  
- **PCA Loadings and Filtering:**  
  Displays the PCA loadings and filters variables with high loadings for each principal component.

## Requirements

- Python 3.6 or higher
- Required Python packages:
  - pandas
  - numpy
  - scikit-learn

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/runciter2078/PCA.git
   ```

2. (Optional) Rename the repository folder to `PCA_SPY` for clarity.

3. Navigate to the project directory:

   ```bash
   cd PCA
   ```

## Usage

1. Ensure that the CSV file (`SPYV3.csv`) is in the project directory.
2. Run the script:

   ```bash
   python pca_analysis.py
   ```

The script will:
- Load and preprocess the dataset.
- Apply scaling using MinMaxScaler.
- Perform PCA on the scaled data.
- Print the explained variance ratios, cumulative variance, and PCA loadings.
- Filter and display variables with high contributions in each principal component.

## Notes

- **Scaling Option:**  
  By default, the script uses MinMax scaling for PCA. To use Standard scaling instead, replace `data_minmax` with `data_standard` when calling the PCA estimator.
  
- **Thresholds for Filtering:**  
  The thresholds for filtering the PCA loadings (e.g., 0.10 for PC-1, 0.15 for PC-2, etc.) are examples and may need adjustment depending on the dataset.

## License

This project is distributed under the [MIT License](LICENSE).
