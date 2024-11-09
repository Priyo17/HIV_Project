import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Load your training data (replace 'HIV_dataset.csv' with your actual dataset)
data = pd.read_csv('HIV_dataset.csv')  # Adjust the path if necessary

# Print out the original column names for debugging
print("Original Columns in dataset:", data.columns.tolist())

# Clean column names (this step ensures consistency across the pipeline)
data.columns = [col.strip().replace('-', '_').replace(' ', '_') for col in data.columns]

# Print out the cleaned column names for debugging
print("Cleaned Columns in dataset:", data.columns.tolist())

# Specify the columns you want to use for training
# Check if these columns exist after cleaning
features_columns = ['Age', 'Marital_Status', 'STD', 'Educational_Background', 
                    'HIV_TEST_IN_PAST_YEAR', 'AIDS_education', 
                    'Places_of_seeking_sex_partners', 'SEXUAL_ORIENTATION', 
                    'Drug_taking']

# Validate feature columns
missing_columns = [col for col in features_columns if col not in data.columns]
if missing_columns:
    raise KeyError(f"The following columns are missing from the dataset: {missing_columns}")

# Select features
features = data[features_columns]

# One-hot encode categorical features
features_encoded = pd.get_dummies(features, drop_first=True)

# Scale the features
scaler = StandardScaler()
scaler.fit(features_encoded)

# Save the fitted scaler
joblib.dump(scaler, 'scaler.pkl')

print("Scaler saved as scaler.pkl")
