import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('tuned_deep_learning_model.keras')

# Load the StandardScaler
scaler = joblib.load('scaler.joblib')

st.title('Diabetes Prediction App')
st.write('Enter patient details to predict diabetes status (0: No Diabetes, 1: Pre-diabetes, 2: Diabetes).')

# Define the order of columns as expected by the model
# This should match X_train.columns from the preprocessing step
# Manually reconstructing based on X_train.head() and df_encoded.info()
# Note: Boolean columns from one-hot encoding will be converted to int/float for prediction
feature_columns = ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
                   'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
                   'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost',
                   'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex',
                   'GenHlth_1.0', 'GenHlth_2.0', 'GenHlth_3.0', 'GenHlth_4.0', 'GenHlth_4.5',
                   'Age_1.0', 'Age_2.0', 'Age_3.0', 'Age_4.0', 'Age_5.0', 'Age_6.0', 'Age_7.0',
                   'Age_8.0', 'Age_9.0', 'Age_10.0', 'Age_11.0', 'Age_12.0', 'Age_13.0',
                   'Education_1.0', 'Education_2.0', 'Education_3.0', 'Education_4.0',
                   'Education_5.0', 'Education_6.0',
                   'Income_1.0', 'Income_2.0', 'Income_3.0', 'Income_4.0',
                   'Income_5.0', 'Income_6.0', 'Income_7.0', 'Income_8.0']

# Streamlit form for user input
with st.form('prediction_form'):
    st.header('Patient Information')

    # Numerical inputs that were scaled
    bmi = st.slider('BMI', 12.0, 98.0, 27.0)
    ment_hlth = st.slider('Mental Health (days of poor mental health)', 0, 30, 0)
    phys_hlth = st.slider('Physical Health (days of poor physical health)', 0, 30, 0)

    # Binary inputs (0 or 1)
    high_bp = st.selectbox('High Blood Pressure', [0, 1])
    high_chol = st.selectbox('High Cholesterol', [0, 1])
    chol_check = st.selectbox('Cholesterol Check within last 5 years', [0, 1])
    smoker = st.selectbox('Smoker', [0, 1])
    stroke = st.selectbox('Had a Stroke', [0, 1])
    heart_disease = st.selectbox('Coronary Heart Disease or Myocardial Infarction', [0, 1])
    phys_activity = st.selectbox('Physical Activity in past 30 days', [0, 1])
    fruits = st.selectbox('Consume Fruit 1 or more times per day', [0, 1])
    veggies = st.selectbox('Consume Vegetables 1 or more times per day', [0, 1])
    hvy_alcohol_consump = st.selectbox('Heavy Alcohol Consumption (adult men >14 drinks/week, adult women >7 drinks/week)', [0, 1])
    any_healthcare = st.selectbox('Have any health care coverage', [0, 1])
    no_doc_bc_cost = st.selectbox('Could not see a doctor because of cost', [0, 1])
    diff_walk = st.selectbox('Difficulty Walking or Climbing Stairs', [0, 1])
    sex = st.selectbox('Sex (0: Female, 1: Male)', [0, 1])

    # Categorical inputs that were one-hot encoded (GenHlth, Age, Education, Income)
    gen_hlth_mapping = {1.0: 'Excellent', 2.0: 'Very good', 3.0: 'Good', 4.0: 'Fair', 4.5: 'Poor (capped)'}
    gen_hlth = st.selectbox('General Health', list(gen_hlth_mapping.keys()), format_func=lambda x: gen_hlth_mapping[x])

    age_mapping = {1.0: '18-24', 2.0: '25-29', 3.0: '30-34', 4.0: '35-39', 5.0: '40-44',
                   6.0: '45-49', 7.0: '50-54', 8.0: '55-59', 9.0: '60-64', 10.0: '65-69',
                   11.0: '70-74', 12.0: '75-79', 13.0: '80+'}
    age = st.selectbox('Age', list(age_mapping.keys()), format_func=lambda x: age_mapping[x])

    education_mapping = {1.0: 'Never attended school or only kindergarten', 2.0: 'Grades 1-8 (Elementary)',
                         3.0: 'Grades 9-11 (Some High School)', 4.0: 'Grade 12 or GED (High School Graduate)',
                         5.0: 'College 1-3 years (Some College or Technical School)',
                         6.0: 'College 4 years or more (College Graduate)'}
    education = st.selectbox('Education Level', list(education_mapping.keys()), format_func=lambda x: education_mapping[x])

    income_mapping = {1.0: '<$10,000', 2.0: '$10,000-$14,999', 3.0: '$15,000-$19,999', 4.0: '$20,000-$24,999',
                      5.0: '$25,000-$34,999', 6.0: '$35,000-$49,999', 7.0: '$50,000-$74,999', 8.0: '$75,000+'}
    income = st.selectbox('Income Level', list(income_mapping.keys()), format_func=lambda x: income_mapping[x])

    submitted = st.form_submit_button('Predict Diabetes')

    if submitted:
        # Create a dictionary from inputs
        user_input_dict = {
            'HighBP': high_bp,
            'HighChol': high_chol,
            'CholCheck': chol_check,
            'BMI': bmi,
            'Smoker': smoker,
            'Stroke': stroke,
            'HeartDiseaseorAttack': heart_disease,
            'PhysActivity': phys_activity,
            'Fruits': fruits,
            'Veggies': veggies,
            'HvyAlcoholConsump': hvy_alcohol_consump,
            'AnyHealthcare': any_healthcare,
            'NoDocbcCost': no_doc_bc_cost,
            'MentHlth': ment_hlth,
            'PhysHlth': phys_hlth,
            'DiffWalk': diff_walk,
            'Sex': sex
        }

        # Add one-hot encoded columns for GenHlth, Age, Education, Income
        for col_prefix, selected_val in [('GenHlth', gen_hlth), ('Age', age), ('Education', education), ('Income', income)]:
            for i in range(1, 14): # Max categories based on Age (13) and Income (8), Education (6), GenHlth (4.5)
                col_name = f'{col_prefix}_{float(i)}.0'
                if col_name in feature_columns: # Only add if it was in the original feature columns
                    user_input_dict[col_name] = 1 if float(i) == selected_val else 0
            # Handle GenHlth_4.5 separately if needed (as it's not an integer)
            if col_prefix == 'GenHlth':
                col_name_4_5 = 'GenHlth_4.5'
                if col_name_4_5 in feature_columns: # Ensure the column exists
                    user_input_dict[col_name_4_5] = 1 if 4.5 == selected_val else 0

        # Create a DataFrame from the user input
        input_df = pd.DataFrame([user_input_dict])

        # Ensure all expected feature columns are present and in the correct order
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0 # Add missing one-hot encoded columns as 0
        input_df = input_df[feature_columns] # Reorder columns to match training data

        # Identify columns to scale (BMI, MentHlth, PhysHlth)
        cols_to_scale = ['BMI', 'MentHlth', 'PhysHlth']

        # Apply scaler to the numerical features
        input_df[cols_to_scale] = scaler.transform(input_df[cols_to_scale])

        # Make prediction
        prediction = model.predict(input_df)
        predicted_class = tf.argmax(prediction, axis=1).numpy()[0]

        st.success(f'Predicted Diabetes Status: {predicted_class}')

st.sidebar.markdown("""
**How to run this application:**

1. Save the code above as `app.py`.
2. Open a terminal or command prompt.
3. Navigate to the directory where `app.py` is saved.
4. Run the command: `streamlit run app.py`
5. The application will open in your web browser.
""")
