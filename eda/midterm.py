
##**2.1 Acquire, Clean, and Preprocess Data**
"""
(a) Data Acquisition
- Identify your data source: file-based (CSV, JSON), database, API, etc.
- Document how you obtained it. For example, if from an API, show the request.
"""

#!pip install kagglehub

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split

import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import os
# Download latest version
file_path = kagglehub.dataset_download("abdallaahmed77/healthcare-risk-factors-dataset")

print("Path to dataset files:", file_path)
# !ls {file_path}

"""##Check if file id loaded correctly."""

csv_path = os.path.join(file_path, "dirty_v3_path.csv")
df = pd.read_csv(csv_path)
print(df.head())

"""(b) Data Cleaning
- Tasks: Handle missing values, remove duplicates, correct invalid entries.
- Python Tools: pandas methods (isnull, dropna, fillna, etc.).
- Tips: Always justify your decisions, e.g., why dropping vs. imputing missing values.
"""

# 1. Check for missing values II
print("Missing values per column:")
print(df.isnull().sum())

# 2. Identify which columns actually have missing data
missing_data_col = df.columns[df.isnull().sum() > 0]
print("\nColumns with missing values:", list(missing_data_col))

# 3. Check for duplicates
print("\nAny duplicated rows?:", df.duplicated().any())

# 4. Show duplicate rows (if any)
duplicates = df[df.duplicated(keep=False)]
print("\nDuplicate rows:")
print(duplicates)

# 5. Drop rows with missing values
# medical condition is a highly related feature, it doesn't has any pattern, imputing missing data might impact the resualt accurate.
df = df.dropna()

# 6. Confirm it's clean
print("\nAfter cleaning:")
print(df.isnull().sum())
print("Shape after cleaning:", df.shape)

# Optional: show info about types
print("\nData types after cleaning:")
print(df.info())

"""(c) Data Preprocessing
- Requirement: Use at least 2 preprocessing techniques (scaling, encoding, feature engineering, etc.).
- Tips: Ensure numeric vs. categorical variables are appropriately transformed.

"""

# Choose your approach: 'tfidf' or 'count'
approach = 'tfidf'  # Change to 'count' for simple word counts

if approach == 'tfidf':
    # TF-IDF: Better for capturing important words across documents
    vectorizer = TfidfVectorizer(
        max_features=50,  # Keep top 50 most important terms
        stop_words='english',  # Remove common words
        ngram_range=(1, 2),  # Include single words and two-word phrases
        min_df=10  # Word must appear in at least 10 titles
    )
else:
    # Count Vectorizer: Simple binary or count representation
    vectorizer = CountVectorizer(
        max_features=50,
        stop_words='english',
        ngram_range=(1, 2),
        binary=True,  # Set to False for counts instead of binary
        min_df=10
    )

# Transform medical condition into features
condition_features = vectorizer.fit_transform(df['Medical Condition'].fillna(''))
condition_feature_names = vectorizer.get_feature_names_out()

# Create DataFrame with medical condition features
title_df = pd.DataFrame(
    condition_features.toarray(),
    columns=[f'condition_{name}' for name in condition_feature_names],
    index=df.index
)
print(title_df.head())

df['BMI_BP_Interaction'] = df['BMI'] * df['Blood Pressure']
df['Age_Stress_Interaction'] = df['Age'] * df['Stress Level']
df['Glucose_per_BMI'] = df['Glucose'] / (df['BMI'] + 1)

df[['BMI_BP_Interaction', 'Age_Stress_Interaction', 'Glucose_per_BMI']].head()

"""##**2.2 Perform Exploratory Data Analysis (EDA) and Visualize Key Insights**

(a) Exploratory Data Analysis
- Compute basic stats (mean, median, std, etc.).
- Identify correlations, outliers, or data imbalances.
- Use pandas describe(), info(), corr() for an overview.
"""

print("---Dataset Overview---")
print(df.info())
#basic stats (mean,median,std)
print(df.describe())

print("---Date Correlation---")
print(df.select_dtypes(exclude=['object', 'string']).corr())

df['Medical Condition'].value_counts()

gender_count = df['Gender'].value_counts()
gender_percent = df['Gender'].value_counts(normalize=True) * 100
gender_summary = pd.DataFrame({
    'Count': gender_count,
    'Percentage': gender_percent.round(2)
})
print(gender_summary)

#Identify correlations, outliers, or data imbalances.
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
analysis_df = pd.concat([df[numeric_cols], title_df], axis=1)

correlation_matrix = analysis_df.corr()

target_cols = ['Glucose', 'Blood Pressure']
health_correlations = correlation_matrix[target_cols].drop(target_cols)

top_n = 20
top_features = health_correlations['Glucose'].abs().nlargest(top_n).index.tolist()

original_numeric = [col for col in numeric_cols if col not in target_cols]
features_to_plot = list(set(original_numeric + top_features))

plot_matrix = correlation_matrix.loc[
    features_to_plot + target_cols,
    features_to_plot + target_cols
]


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 25))

sns.heatmap(
    plot_matrix,
    annot=True,
    cmap='coolwarm',
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8},
    fmt='.2f',
    vmin=-1, vmax=1,
    ax=ax1
)
ax1.set_title('Correlation Matrix (Health + Condition Features)', fontsize=12, fontweight='bold')
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=8)
ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0, fontsize=8)

focus_plot = health_correlations.loc[features_to_plot].sort_values('Glucose', key=abs, ascending=False)
sns.heatmap(
    focus_plot,
    annot=True,
    cmap='coolwarm',
    center=0,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8},
    fmt='.2f',
    vmin=-1, vmax=1,
    ax=ax2
)
ax2.set_title('Feature Correlations with Glucose and Blood Pressure', fontsize=12, fontweight='bold')
ax2.set_xticklabels(['Glucose', 'Blood Pressure'], rotation=0)
ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0, fontsize=8)

plt.tight_layout()
plt.show()


print("\n" + "="*80)
print("TOP CONDITION FEATURES CORRELATED WITH HEALTH OUTCOMES")
print("="*80)

for target in target_cols:
    print(f"\n{target.upper()}:")
    print("-" * 80)

    cond_corr = health_correlations[target][
        health_correlations.index.str.startswith('condition_')
    ].sort_values(key=abs, ascending=False).head(15)

    print(f"{'Feature':<40} {'Correlation':>12} {'Interpretation':>20}")
    print("-" * 80)

    for feature, corr in cond_corr.items():
        word = feature.replace('condition_', '')
        interpretation = "Higher with increasing values" if corr > 0 else "Decreases as values rise"
        print(f"{word:<40} {corr:>12.4f} {interpretation:>20}")


print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)
print(f"Total condition features created: {len(condition_feature_names)}")
print(f"Features in visualization: {len(features_to_plot)}")

strongest_pos = health_correlations['Glucose'].nlargest(1)
strongest_neg = health_correlations['Glucose'].nsmallest(1)

print(f"\nStrongest positive correlation with Glucose:")
print(f"  {strongest_pos.index[0]}: {strongest_pos.values[0]:.4f}")

print(f"\nStrongest negative correlation with Glucose:")
print(f"  {strongest_neg.index[0]}: {strongest_neg.values[0]:.4f}")

# Select numeric columns for correlation analysis
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Calculate correlation matrix
correlation_matrix = df[numeric_cols].corr()

# Create figure with larger size for better readability
plt.figure(figsize=(15, 15))

# Create heatmap
sns.heatmap(correlation_matrix,
            annot=True,  # Show correlation values
            cmap='coolwarm',  # Color scheme: blue for negative, red for positive
            center=0,  # Center colormap at 0
            square=True,  # Make cells square-shaped
            linewidths=1,  # Add gridlines
            cbar_kws={"shrink": 0.8},  # Adjust colorbar size
            fmt='.2f',  # Format numbers to 2 decimal places
            vmin=-1, vmax=1)  # Set color scale from -1 to 1

plt.title('Correlation Matrix - Identifying Stay Predictors',
          fontsize=14, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

#Identify the pattern between Length of Stay with all other numberical featueres

key_features = [
    'Age','Glucose','Blood Pressure','BMI','Oxygen Saturation', 'Cholesterol','Triglycerides','HbA1c','Diet Score',
    'Stress Level', 'Physical Activity', 'Age_Stress_Interaction', 'Sleep Hours'
]

plt.figure(figsize=(15, 10))
for i, feature in enumerate(key_features, 1):
    plt.subplot(4, 4, i)
    sns.scatterplot(data=df, x=feature, y='LengthOfStay', alpha=0.6)
    plt.title(f'Length of Stay vs {feature}')
    plt.xlabel(feature)
    plt.ylabel('Length of Stay')

plt.suptitle('Length of Stay vs Key Health Indicators', fontsize=16, fontweight='bold', y=1)
plt.tight_layout()
plt.show()

"""(b) Data Visualization
- Requirement: At least 3 different visualization techniques (histogram, scatter plot, box plot, heatmap, etc.).
- Tips: Use clear labels, titles, and legends. Let visuals drive your EDA narrative.

"""

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", palette="pastel")

plt.figure(figsize=(8, 5))
sns.histplot(df['Glucose'], bins=30, kde=True, color='pink')
plt.title('Distribution of Glucose Levels', fontsize=14, fontweight='bold')
plt.xlabel('Glucose')
plt.ylabel('Frequency')
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Medical Condition', y='LengthOfStay', palette='Set2')
plt.title('Length of Stay by Medical Condition', fontsize=14, fontweight='bold')
plt.xlabel('Medical Condition')
plt.ylabel('Length of Stay (days)')
plt.xticks(rotation=30, ha='right')
plt.show()


plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df, x='BMI', y='Blood Pressure',
    hue='Stress Level', palette='coolwarm', alpha=0.7
)
plt.title('Relationship Between BMI, Blood Pressure, and Stress Level', fontsize=14, fontweight='bold')
plt.xlabel('BMI')
plt.ylabel('Blood Pressure')
plt.legend(title='Stress Level')
plt.show()

"""##**2.3 Build and Evaluate a Machine Learning Model**

(a) Model Building
- Requirement: At least 2 different ML algorithms (e.g., Logistic Regression, Random Forest, Linear Regression, etc.).
- Tips: Match the algorithm type to your target variable (classification vs. regression).

#Random Forest
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pickle

df_condition = pd.DataFrame(condition_features)

# Define features (X) and target (y)
# Drop the target variable and the original categorical columns
X = df.drop(columns=[
    'LengthOfStay',
    'Gender',
    'Medical Condition',
    'random_notes',
    'noise_col',
    'BMI',
    'Diet Score',
  ])

X = pd.concat([X, title_df], axis=1)

y = df['LengthOfStay']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Model Selection: Random Forest Regressor
# A robust non-linear model that can capture complex relationships.
#model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10, min_samples_leaf=5)
#optimize the model using n_estimators = 500, max_depth = 5
model = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1, max_depth=5, min_samples_leaf=5)

# Train the model
print("Training Random Forest Regressor...")
model.fit(X_train, y_train)
print("Training complete.")

# 2. Prediction and Evaluation
y_pred = model.predict(X_test)

# Since LengthOfStay must be an integer, we round the predictions
y_pred_rounded = np.round(y_pred)

# Evaluation Metrics
mae = mean_absolute_error(y_test, y_pred_rounded)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_rounded))
r2 = r2_score(y_test, y_pred_rounded)

print("\n--- Model Evaluation (Random Forest Regressor) ---")
print(f"Mean Absolute Error (MAE): {mae:.4f} days")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f} days")
print(f"R-squared (R2): {r2:.4f}")

"""
test the best parameter value for random forest model
"""
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import r2_score
# from sklearn.model_selection import train_test_split
# import pandas as pd

# # Define parameter ranges to test
# n_estimators_list = [50, 100, 200, 500]
# max_depth_list = [5, 10, 20, None]

# results = []

# # Loop through combinations
# for n in n_estimators_list:
#     for d in max_depth_list:
#         model = RandomForestRegressor(
#             n_estimators=n,
#             max_depth=d,
#             min_samples_leaf=5,
#             random_state=42,
#             n_jobs=-1
#         )

#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_test)
#         r2 = r2_score(y_test, y_pred)

#         results.append({
#             'n_estimators': n,
#             'max_depth': d,
#             'r2_score': r2
#         })
#         print(f"n_estimators={n}, max_depth={d}, R²={r2:.4f}")

# 3. Feature Importance Analysis
# This will show which features, including the Medical Condition dummies, are most influential.
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
rf_top_10_features = feature_importances.nlargest(10)

print("\n--- Random Forest Model Top 10 Feature Importances ---")
print(rf_top_10_features.to_markdown(numalign="left", stralign="left"))

# Save the feature importances to a file for documentation
with open('feature_importances.txt', 'w') as f:
    f.write(rf_top_10_features.to_markdown(numalign="left", stralign="left"))

# Save the trained model (optional, but good practice)
with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\nModel training and evaluation complete. Results saved.")

"""#Linear Regression"""

from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

imputer = SimpleImputer(strategy='median')
X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

y_pred_rf = model.predict(X_test_imputed )
y_pred_rf_rounded = np.round(y_pred_rf)

lr_model = LinearRegression()
lr_model.fit(X_train_imputed , y_train)
y_pred_lr = lr_model.predict(X_test_imputed )


lr_mae = mean_absolute_error(y_test, y_pred_lr)
lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))
lr_r2 = r2_score(y_test, y_pred_lr)

print("lr_mae:", lr_mae)
print("lr_rmse:", lr_rmse)
print("lr_r2:", lr_r2)

feature_importances = pd.Series(
    np.abs(lr_model.coef_),  # use absolute value of coefficients
    index=X.columns
)

lr_top_10_features = feature_importances.nlargest(10)

print("--Linear Regression Model Top 10 Important Features--")
print(lr_top_10_features)

"""#XGBoost"""

from xgboost import XGBRegressor

# 2. Model Selection: XGBoost Regressor
# A gradient boosting model that can provide good performance.
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1)

# Train the model
print("\nTraining XGBoost Regressor...")
xgb_model.fit(X_train_imputed, y_train)
print("Training complete.")

# 3. Prediction and Evaluation
y_pred_xgb = xgb_model.predict(X_test_imputed)

# Since LengthOfStay must be an integer, we round the predictions
y_pred_xgb_rounded = np.round(y_pred_xgb)

# Evaluation Metrics
xgb_mae = mean_absolute_error(y_test, y_pred_xgb_rounded)
xgb_rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgb_rounded))
xgb_r2 = r2_score(y_test, y_pred_xgb_rounded)

print("\n--- Model Evaluation (XGBoost Regressor) ---")
print(f"Mean Absolute Error (MAE): {xgb_mae:.4f} days")
print(f"Root Mean Squared Error (RMSE): {xgb_rmse:.4f} days")
print(f"R-squared (R2): {xgb_r2:.4f}")

# Feature Importance Analysis
xgb_feature_importances = pd.Series(xgb_model.feature_importances_, index=X.columns)
xgb_top_10_features = xgb_feature_importances.nlargest(10)

print("\n--- XGBoost Model Top 10 Feature Importances ---")
print(xgb_top_10_features.to_markdown(numalign="left", stralign="left"))

# Compare with other models
print("\n--- Model Evaluation Comparison ---")
print(f"{'Model':<20}{'MAE':<15}{'RMSE':<15}{'R²':<15}")
print("-"*65)
print(f"{'Random Forest':<20}{mae:<15.4f}{rmse:<15.4f}{r2:<15.4f}")
print(f"{'Linear Regression':<20}{lr_mae:<15.4f}{lr_rmse:<15.4f}{lr_r2:<15.4f}")
print(f"{'XGBoost':<20}{xgb_mae:<15.4f}{xgb_rmse:<15.4f}{xgb_r2:<15.4f}")


"""(b) Model Evaluation
- Requirement: At least 2 different evaluation metrics (accuracy, precision/recall, F1, RMSE, MAE, etc.).
- Tips: Present numeric results and interpret them in plain English.
Consider basic hyperparameter tuning.

"""

import pandas as pd
import matplotlib.pyplot as plt

print("\n=== Model Evaluation Dashboard ===")

# 1. Model performance summary table
dashboard_metrics = pd.DataFrame({
    'Model': ['Random Forest', 'Linear Regression', 'XGBoost'],
    'MAE (avg days off)': [mae, lr_mae, xgb_mae],
    'RMSE (error size)': [rmse, lr_rmse, xgb_rmse],
    'R² (variance)': [r2, lr_r2, xgb_r2]
})

print("\nModel Performance Summary:")
display(dashboard_metrics.round(3))  

#2. Top 10 important features comparison

print("\n===Top 10 Important Features Comparison===")

df_importance = pd.DataFrame({
    'Random Forest': rf_top_10_features.index,
    'Linear Regression': lr_top_10_features.index,
    'XGBoost': rf_top_10_features.index
})
print(df_importance)

