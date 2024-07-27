

import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the model and dataset
with open('disease_pred.pkl', 'rb') as f:
    model = pickle.load(f)
df = pd.read_csv('Training.csv')
df.drop('Unnamed: 133', axis=1, inplace=True)

# Define the map_symptoms function
def map_symptoms(symptoms):
    dataset_symptoms = df.columns.tolist()
    dataset_symptoms.pop()
    mapped_symptoms = [0] * len(dataset_symptoms)
    for symptom in symptoms:
        symptom = symptom.strip().replace(' ', '_')
        if symptom in dataset_symptoms:
            mapped_symptoms[dataset_symptoms.index(symptom)] = 1
    return mapped_symptoms

# Set the title
st.title("Disease Prediction Model")

# Instructions
st.write("Please select up to 5 symptoms from the dropdowns below and click Analyze to predict the disease.")

# List of example symptoms
symptoms = [
    "abdominal_pain", "acidity", "acute_liver_failure", "altered_sensorium", "anxiety", "back_pain", "belly_pain", "blackheads", "bloody_stool", "blister", "blurred_and_distorted_vision", "brittle_nails", "bruising", "burning_micturition", "chills", "chest_pain", "coma", "congestion", "continuous_feel_of_urine", "continuous_sneezing", "constipation", "cramps", "dark_urine", "dehydration", "depression", "dizziness", "distention_of_abdomen", "distorted_vision", "diarrhoea", "dischromic_patches", "drying_and_tingling_lips", "enlarged_thyroid", "excessive_hunger", "extra_marital_contacts", "family_history", "fast_heart_rate", "fatigue", "fluid_overload", "foul_smell_of_urine", "headache", "hip_joint_pain", "history_of_alcohol_consumption", "hypertension", "inflammatory_nails", "indigestion", "internal_itching", "irritability", "irritation_in_anus", "irregular_sugar_level", "itching", "joint_pain", "knee_pain", "lack_of_concentration", "lethargy", "loss_of_appetite", "loss_of_balance", "loss_of_smell", "malaise", "mild_fever", "mood_swings", "movement_stiffness", "mucoid_sputum", "muscle_pain", "muscle_wasting", "muscle_weakness", "nausea", "neck_pain", "nodal_skin_eruptions", "obesity", "pain_behind_the_eyes", "pain_during_bowel_movements", "pain_in_anal_region", "painful_walking", "palpitations", "passage_of_gases", "patches_in_throat", "phlegm", "polyuria", "prominent_veins_on_calf", "puffy_face_and_eyes", "pus_filled_pimples", "red_sore_around_nose", "red_spots_over_body", "restlessness", "rusty_sputum", "runny_nose", "shivering", "silver_like_dusting", "sinus_pressure", "skin_peeling", "skin_rash", "slurred_speech", "small_dents_in_nails", "spinning_movements", "spotting_urination", "stiff_neck", "stomach_bleeding", "stomach_pain", "sweating", "swelling_extremeties", "swelling_joints", "swelling_of_stomach", "swollen_blood_vessels", "swollen_legs", "toxic_look_(typhos)", "throat_irritation", "throat_pain", "unsteadiness", "ulcers_on_tongue", "visual_disturbances", "vomiting", "weakness_in_limbs", "weakness_of_one_body_side", "weight_gain", "weight_loss", "yellow_crust_ooze", "yellow_urine", "yellowing_of_eyes"
]


# Dropdowns for symptoms
symptom1 = st.selectbox(label="Select Symptom 1", options=symptoms)
symptom2 = st.selectbox(label="Select Symptom 2", options=symptoms)
symptom3 = st.selectbox(label="Select Symptom 3", options=symptoms)
symptom4 = st.selectbox(label="Select Symptom 4", options=symptoms)
symptom5 = st.selectbox(label="Select Symptom 5", options=symptoms)


# Analyze button
if st.button("Analyze"):
    # List of selected symptoms
    selected_symptoms = [symptom1, symptom2, symptom3, symptom4, symptom5]
    
    # Remove empty selections
    selected_symptoms = [symptom for symptom in selected_symptoms if symptom]
    
    # Check if at least one symptom is selected
    if len(selected_symptoms) < 1:
        st.error("Please select at least one symptom.")
    else:
        # Map symptoms to dataset format
        mapped_symptoms = map_symptoms(selected_symptoms)
        
        # Make prediction
        s = np.array(mapped_symptoms).reshape(1, -1)
        prediction = model.predict(s)
        predicted_disease = prediction[0]
        
        # Display the result in a modal
        st.success(f"The predicted disease is: {predicted_disease}")

else:
    st.error("Please select at least one symptom and click Analyze.")
