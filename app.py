import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import streamlit as st

# Load your trained model and the scaler
model = keras.models.load_model('HIV_Prediction.h5')
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
scaler = joblib.load('scaler.pkl')

# Define expected columns with normalized names (lowercase, consistent formatting)
expected_columns = ['age', 'marital_status_married', 'marital_status_unmarried',
                    'std_no', 'std_yes', 'educational_background_college_degree', 
                    'educational_background_post_graduate', 'educational_background_senior_high_school',
                    'hiv_test_in_past_year_no', 'hiv_test_in_past_year_yes',
                    'aids_education_no', 'aids_education_yes',
                    'places_of_seeking_sex_partners_bar', 'places_of_seeking_sex_partners_internet',
                    'places_of_seeking_sex_partners_park', 'sexual_orientation_bisexual',
                    'sexual_orientation_heterosexual', 'drug_taking_no', 'drug_taking_yes']

# Function to preprocess data
def preprocess_data(data):
    # Standardize column names in uploaded data (lowercase, replace spaces/hyphens)
    data.columns = data.columns.str.strip().str.replace(' ', '_').str.replace('-', '_').str.lower()

    # One-hot encode the data
    data_encoded = pd.get_dummies(data, drop_first=True)

    # Ensure missing columns are filled with 0
    for col in expected_columns:
        if col not in data_encoded.columns:
            data_encoded[col] = 0

    # Reorder columns to match expected order
    data_encoded = data_encoded[expected_columns]

    # Apply scaling
    return scaler.transform(data_encoded)

# Function to make predictions
def make_prediction(data):
    processed_data = preprocess_data(data)
    return model.predict(processed_data)

# Enhance UI with background image using custom CSS
def set_background(image_file):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url({image_file});
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# UI for the app
def render_home_page():
    set_background('https://example.com/background_image.jpg')  # Replace with your image URL
    st.title("🧬 HIV Prediction App")
    st.write("This app predicts HIV risk based on various features. Upload a CSV file containing relevant data to get predictions.")
    st.markdown(
        """
        #### Instructions:
        1. Ensure the file is in CSV format.
        2. Data should include columns such as Age, Marital Status, STD, and more.
        """
    )
    
    # File upload section
    uploaded_file = st.file_uploader("Upload a CSV file", type='csv')
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
        
        with st.expander("Preview of Uploaded Data"):
            st.write(data.head())
        
        if st.button("Make Prediction"):
            st.session_state['data'] = data
            st.session_state['page'] = 'prediction'

def render_prediction_page():
    st.title("Prediction Results")
    
    if 'data' in st.session_state:
        data = st.session_state['data']
        predictions = make_prediction(data)
        
        # Combine input data with predictions
        results_df = data.copy()
        results_df['Prediction'] = predictions.flatten()
        
        st.write("Predictions with Input Data:")
        st.dataframe(results_df)
        
        # Download option
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions as CSV", csv, "predictions.csv", "text/csv")
        
        if st.button("Go Back"):
            st.session_state['page'] = 'home'
    else:
        st.warning("No data uploaded. Please upload data first.")
        if st.button("Go Back"):
            st.session_state['page'] = 'home'

# Page Navigation
if 'page' not in st.session_state:
    st.session_state['page'] = 'home'

if st.session_state['page'] == 'home':
    render_home_page()
elif st.session_state['page'] == 'prediction':
    render_prediction_page()
