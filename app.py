import streamlit as st
import pickle
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sqlite3 


# connect to db
conn = sqlite3.connect("final_data.db")
cursor = conn.cursor()


# Page Configuration
st.set_page_config(
    page_title="Readmission Risk Predictor",
    layout="wide"
)

# Styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Section headers */
.section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.70rem;
    text-transform: uppercase;
    letter-spacing: 4px;
    color: #0e7490;
    margin: 0 0 10px 0;
    border-bottom: 2px solid #0e7490;
    padding-bottom: 6px;
}

/* Spacer to push sections into alignment */
.spacer-sm  { margin-top: 14px; }
.spacer-md  { margin-top: 98px; }
.spacer-lg  { margin-top: 97px; }

/* Risk boxes */
.risk-box {
    border-radius: 14px;
    padding: 12px;
    text-align: center;
    margin-bottom: 16px;
}
.risk-high {
    background: linear-gradient(135deg, #3d0000, #7a0000);
    border: 1px solid #ff4444;
    color: #ffcccc;
}
.risk-low {
    background: linear-gradient(135deg, #003d1a, #005c28);
    border: 1px solid #00cc66;
    color: #ccffe0;
}
.risk-score {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 4rem;
    font-weight: 700;
    display: block;
    margin: 6px 0;
    line-height: 1;
}
.risk-label {
    font-size: 1rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 3px;
}
.risk-sub {
    font-size: 0.85rem;
    opacity: 0.8;
    margin-top: 6px;
}

/* Driver rows */
.driver-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #f8f9fa;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 10px 16px;
    margin: 5px 0;
    font-size: 0.88rem;
    color: #1a1a1a;
}
.driver-name {color: #1a1a1a; font-weight: 500;}
.driver-up {color: #c0392b; font-weight: 600; white-space: nowrap; margin-left: 12px;}
.driver-down {color: #1a7a3f; font-weight: 600; white-space: nowrap; margin-left: 12px;}

/* Submit button */
div[data-testid="stFormSubmitButton"] {
    display: flex !important;
    justify-content: center !important;
    margin-top: 32px;
}
div[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, #0e7490, #0e7490) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 79px !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    min-width: 360px;
}
div[data-testid="stFormSubmitButton"] > button:hover {
    opacity: 0.88 !important;
}
</style>
""", unsafe_allow_html=True)


# Load model
@st.cache_resource
def load_model():
    with open('gb_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('feature_columns.pkl', 'rb') as f:
        columns = pickle.load(f)
    explainer = shap.TreeExplainer(model)
    return model, columns, explainer

model, feature_columns, explainer = load_model()

# Reference data
diag_categories = ["Select"] + sorted([
    "Complications Of Pregnancy, Childbirth, And The Puerperium",
    "Diseases Of The Circulatory System",
    "Diseases Of The Digestive System",
    "Diseases Of The Genitourinary System",
    "Diseases Of The Musculoskeletal System And Connective Tissue",
    "Diseases Of The Nervous System And Sense Organs",
    "Diseases Of The Respiratory System",
    "Diseases Of The Skin And Subcutaneous Tissue",
    "Endocrine, Nutritional And Metabolic Diseases, And Immunity Disorders",
    "Infectious And Parasitic Diseases",
    "Injury And Poisoning",
    "Mental Disorders",
    "Neoplasms",
    "Other",
    "Supplementary Classification Of External Causes Of Injury And Poisoning",
    "Supplementary Classification Of Factors Influencing Health Status And Contact With Health Services",
    "Symptoms, Signs, And Ill-Defined Conditions",
])

admission_type = {0: "Select", 1: "Emergency", 2: "Urgent", 3: "Elective", 4: "Newborn", 7: "Trauma Center"}
discharge_disp = {0: "Select", 1: "Home", 2: "Short Term Hospital", 3: "Skilled Nursing Facility", 6: "Home with Home Health", 13: "Hospice / Home"}
admission_source = {0: "Select", 1: "Physician Referral", 2: "Clinic Referral", 3: "HMO Referral", 4: "Transfer from Hospital", 7: "Emergency Room"}


def clean_feature_name(feat):
    return (feat
        .replace('diag_1_category_', 'Primary Dx: ')
        .replace('diag_2_category_', 'Secondary Dx: ')
        .replace('diag_3_category_', 'Tertiary Dx: ')
        .replace('race_', 'Race: ')
        .replace('age_', 'Age Group: ')
        .replace('gender_', 'Gender: ')
        .replace('num_hospitalizations', 'Prior Hospitalizations')
        .replace('discharge_disposition_id', 'Discharge Destination')
        .replace('number_inpatient', 'Prior Inpatient Visits')
        .replace('total_visits', 'Total Prior Visits')
        .replace('admission_source_id', 'Admission Source')
        .replace('time_in_hospital', 'Length of Stay (days)')
        .replace('admission_type_id', 'Admission Type')
        .replace('number_diagnoses', 'Number of Diagnoses')
        .replace('num_medications', 'Number of Medications')
        .replace('avg_procedure', 'Procedures per Day')
        .replace('num_med_changes', 'Medication Changes')
        .replace('num_med_increase', 'Medication Increases')
        .replace('diabetesMed', 'On Diabetes Medication')
        .replace('change', 'Medication Changed')
        .replace('_', ' ')
        .strip()
    )


def build_input(inputs):
    row = {col: 0 for col in feature_columns}
    for key in ['admission_type_id', 'discharge_disposition_id', 'admission_source_id',
                'time_in_hospital', 'num_procedures', 'num_medications',
                'number_outpatient', 'number_emergency', 'number_inpatient',
                'number_diagnoses', 'change', 'diabetesMed', 'num_hospitalizations',
                'avg_procedure', 'total_visits', 'num_med_changes', 'num_med_increase']:
        if key in row:
            row[key] = inputs[key]
    if f"race_{inputs['race']}" in row:
        row[f"race_{inputs['race']}"] = 1
    if inputs['gender'] == 'Male':
        row['gender_Male'] = 1
    if f"age_{inputs['age']}" in row:
        row[f"age_{inputs['age']}"] = 1
    for d in [1, 2, 3]:
        col = f"diag_{d}_category_{inputs[f'diag_{d}']}"
        if col in row:
            row[col] = 1
    return pd.DataFrame([row])[feature_columns]


# Header
st.markdown("# 30-Day Readmission Risk Predictor")
st.caption("Enter patient details below to assess readmission risk and understand the key contributing factors.")
st.markdown("---")

# Input form
with st.form("patient_form"):
    col1, col2, col3 = st.columns(3, gap="large")

    # Demographics and diagnoses
    with col1:
        st.markdown('<p class="section-header">Demographics</p>', unsafe_allow_html=True)
        age = st.selectbox("Age Group", ["Select", "10-19","20-29","30-39","40-49","50-59","60-69","70-79","80-89","90-100"], index=0)
        gender = st.selectbox("Gender", ["Select", "Female", "Male"], index=0)
        race = st.selectbox("Race", ["Select", "AfricanAmerican", "Asian", "Caucasian", "Hispanic", "Other"], index=0)

        st.markdown('<div class="spacer-md"><p class="section-header">Diagnoses</p></div>', unsafe_allow_html=True)
        diag_1 = st.selectbox("Primary Diagnosis", diag_categories, index=0)
        diag_2 = st.selectbox("Secondary Diagnosis", diag_categories, index=0)
        diag_3 = st.selectbox("Tertiary Diagnosis", diag_categories, index=0)

    # Admission Info and prior visits
    with col2:
        st.markdown('<p class="section-header">Admission Info</p>', unsafe_allow_html=True)
        admission_type_id = st.selectbox("Admission Type", options=list(admission_type.keys()), format_func=lambda x: admission_type[x], index=0)
        admission_source_id = st.selectbox("Admission Source", options=list(admission_source.keys()), format_func=lambda x: admission_source[x], index=0)
        discharge_disposition_id = st.selectbox("Discharge Destination", options=list(discharge_disp.keys()), format_func=lambda x: discharge_disp[x], index=0)
        time_in_hospital = st.slider("Days in Hospital", 1, 14, 0)

        st.markdown('<div class="spacer-sm"><p class="section-header">Prior Visits</p></div>', unsafe_allow_html=True)
        num_hospitalizations = st.number_input("Total Prior Hospitalizations", 0, 50, 0)
        number_inpatient = st.number_input("Inpatient Visits (prior year)", 0, 20, 0)
        number_outpatient = st.number_input("Outpatient Visits (prior year)", 0, 40, 0)
        number_emergency = st.number_input("Emergency Visits (prior year)", 0, 30, 0)

        submitted = st.form_submit_button("Predict Readmission Risk")

    # Clinical Metrics and medications
    with col3:
        st.markdown('<p class="section-header">Clinical Metrics</p>', unsafe_allow_html=True)
        num_procedures = st.number_input("Number of Procedures", 0, 20, 0)
        num_medications = st.number_input("Number of Medications", 0, 80, 0)
        number_diagnoses = st.number_input("Number of Diagnoses", 0, 16, 0)

        st.markdown('<div class="spacer-lg"><p class="section-header">Medications</p></div>', unsafe_allow_html=True)
        diabetesMed = st.selectbox("On Diabetes Medication?", options=["Select", "Yes", "No"], index=0)
        change = st.selectbox("Medication Changed During Stay?", options=["Select", "Yes", "No"], index=0)
        num_med_changes = st.number_input("Number of Medication Changes", 0, 20, 0)
        num_med_increase = st.number_input("Number of Medication Increases", 0, 20, 0)


# Results
if submitted:
    missing = []
    if age == "Select": missing.append("Age Group")
    if gender == "Select": missing.append("Gender")
    if race == "Select": missing.append("Race")
    if diag_1 == "Select": missing.append("Primary Diagnosis")
    if diag_2 == "Select": missing.append("Secondary Diagnosis")
    if diag_3 == "Select": missing.append("Tertiary Diagnosis")
    if admission_type_id == 0: missing.append("Admission Type")
    if admission_source_id == 0: missing.append("Admission Source")
    if discharge_disposition_id == 0: missing.append("Discharge Destination")
    if diabetesMed == "Select": missing.append("On Diabetes Medication")
    if change == "Select": missing.append("Medication Changed During Stay")

    if missing:
        st.error(f"Please complete the following fields before predicting: {', '.join(missing)}")
        st.stop()

    avg_procedure = num_procedures/time_in_hospital if time_in_hospital > 0 else 0
    total_visits = number_inpatient + number_outpatient + number_emergency

    inputs = dict(
        admission_type_id=admission_type_id, discharge_disposition_id=discharge_disposition_id,
        admission_source_id=admission_source_id, time_in_hospital=time_in_hospital,
        num_procedures=num_procedures, num_medications=num_medications,
        number_outpatient=number_outpatient, number_emergency=number_emergency,
        number_inpatient=number_inpatient, number_diagnoses=number_diagnoses,
        change=1 if change == "Yes" else 0,
        diabetesMed=1 if diabetesMed == "Yes" else 0,
        num_hospitalizations=num_hospitalizations,
        avg_procedure=avg_procedure, total_visits=total_visits,
        num_med_changes=num_med_changes, num_med_increase=num_med_increase,
        race=race, gender=gender, age=age,
        diag_1=diag_1, diag_2=diag_2, diag_3=diag_3,
    )

    input_df = build_input(inputs)
    prob = model.predict_proba(input_df)[0][1]
    risk_pct = round(prob * 100, 1)
    is_high = prob >= 0.5

    # check if prediction result is saved in database
    query = """SELECT readmission_risk FROM predictions WHERE admission_type_id = :admission_type_id
    AND discharge_disposition_id = :discharge_disposition_id
    AND admission_source_id = :admission_source_id
    AND time_in_hospital = :time_in_hospital
    AND num_procedures = :num_procedures
    AND num_medications = :num_medications
    AND number_outpatient = :number_outpatient
    AND number_emergency = :number_emergency
    AND number_inpatient = :number_inpatient
    AND number_diagnoses = :number_diagnoses
    AND change = :change
    AND diabetesMed = :diabetesMed
    AND num_hospitalizations = :num_hospitalizations
    AND avg_procedure = :avg_procedure
    AND total_visits = :total_visits
    AND num_med_changes = :num_med_changes
    AND num_med_increase = :num_med_increase
    AND race = :race
    AND gender = :gender
    AND age = :age
    AND diag_1 = :diag_1
    AND diag_2 = :diag_2
    AND diag_3 = :diag_3"""
    
    cursor.execute(query, inputs)
    result = cursor.fetchall()

    # if result is not saved, add to database
    if not result:
        print("adding to db")
        query = """INSERT INTO predictions (
        admission_type_id, discharge_disposition_id, admission_source_id, 
        time_in_hospital, num_procedures, num_medications, 
        number_outpatient, number_emergency, number_inpatient, 
        number_diagnoses, change, diabetesMed, num_hospitalizations, 
        avg_procedure, total_visits, num_med_changes, 
        num_med_increase, race, gender, age, diag_1, diag_2, diag_3, readmission_risk
        ) VALUES (
            :admission_type_id, :discharge_disposition_id, :admission_source_id, 
            :time_in_hospital, :num_procedures, :num_medications, 
            :number_outpatient, :number_emergency, :number_inpatient, 
            :number_diagnoses, :change, :diabetesMed, :num_hospitalizations, 
            :avg_procedure, :total_visits, :num_med_changes, 
            :num_med_increase, :race, :gender, :age, :diag_1, :diag_2, :diag_3, :readmission_risk
        )"""
        inputs['readmission_risk']=risk_pct
        cursor.execute(query, inputs)
        conn.commit()

    st.markdown("---")
    st.markdown("## Results")

    r1, r2 = st.columns([1, 1], gap="large")

    with r1:
        box_cls = "risk-high" if is_high else "risk-low"
        label = "High Risk" if is_high else "Low Risk"
        st.markdown(f"""
        <div class="risk-box {box_cls}">
            <span class="risk-label">{label}</span>
            <span class="risk-score">{risk_pct}%</span>
            <span class="risk-sub">probability of 30-day readmission</span>
        </div>""", unsafe_allow_html=True)

        if is_high:
            st.warning("Recommendation: Consider enhanced discharge planning, early follow-up within 7 days, and inform patient about warning signs.")
        else:
            st.success("Recommendation: Standard discharge procedures seem appropriate. Routine follow-up advised.")

        # SHAP Analysis
        st.caption("Red bars push the risk score higher. Blue bars push it lower. Each bar represents one patient factor.")

        shap_explanation = explainer(input_df)

        with plt.style.context('default'):
            shap.plots.waterfall(shap_explanation[0], max_display=10, show=False)
            fig = plt.gcf()
            fig.set_size_inches(8, 4)
            fig.patch.set_facecolor('white')
            for ax_ in fig.axes:
                ax_.set_facecolor('white')
                ax_.tick_params(colors='black', labelsize=8)
                ax_.xaxis.label.set_color('black')
                for spine in ax_.spines.values():
                    spine.set_edgecolor('#cccccc')
            for text_obj in fig.findobj(matplotlib.text.Text):
                text_obj.set_color('black')
                text_obj.set_fontsize(8)
            plt.tight_layout()
            _, mid, _ = st.columns([0.5, 2, 0.5])
            with mid:
                st.pyplot(fig)
            plt.close()

    with r2:
        st.markdown('<p class="section-header">Top Factors Driving This Prediction</p>', unsafe_allow_html=True)
        shap_vals = explainer.shap_values(input_df)[0]
        shap_series = pd.Series(shap_vals, index=feature_columns)
        top = shap_series.abs().nlargest(8).index

        for feat, val in shap_series[top].items():
            name = clean_feature_name(feat)
            if val > 0:
                st.markdown(
                    f'<div class="driver-row">'
                    f'<span class="driver-name">{name}</span>'
                    f'<span class="driver-up">&#9650; Increases risk &nbsp; {val:+.3f}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="driver-row">'
                    f'<span class="driver-name">{name}</span>'
                    f'<span class="driver-down">&#9660; Decreases risk &nbsp; {val:+.3f}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )