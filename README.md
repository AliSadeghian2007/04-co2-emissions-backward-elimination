# 04 - CO2 Emissions: Backward Elimination

Predicting vehicle CO2 emissions with Multiple Linear Regression and
removing redundant features using Backward Elimination (p-value based).

## Dataset
[CO2 Emissions Canada](https://www.kaggle.com/) (Kaggle).
Download `CO2 Emissions_Canada.csv` and place it in the `data/` folder.

## Approach
1. Exploratory data analysis
2. Feature selection and correlation analysis
3. Preprocessing (OneHotEncoder inside ColumnTransformer, split before fit)
4. Baseline LinearRegression with all features
5. Backward Elimination using statsmodels OLS (threshold: 0.05)
6. Retrain and compare with the baseline

## Results
_To be filled after the analysis._

## How to Run
pip install -r requirements.txt
jupyter notebook

## Project Structure
- notebook.ipynb: full analysis
- data/: dataset (not tracked by Git)
- requirements.txt: dependencies