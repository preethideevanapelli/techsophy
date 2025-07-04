import streamlit as st
import pandas as pd

st.set_page_config(page_title="No-Show Predictor Dashboard", layout="wide")
st.title("📅 Appointment No-Show Prediction")

# Load data
df = pd.read_csv("predictions.csv")
df['appointment_day'] = pd.to_datetime(df['appointment_day'])

# Filters
risk_range = st.slider("Filter by No-Show Risk Probability", 0.0, 1.0, (0.5, 1.0), 0.01)
appt_types = st.multiselect("Appointment Types",
                           options=df['appointment_type'].unique(),
                           default=df['appointment_type'].unique())

seasons = st.multiselect("Seasons",
                         options=df['season'].unique(),
                         default=df['season'].unique())

filtered_df = df[
    (df['no_show_prob'].between(risk_range[0], risk_range[1])) &
    (df['appointment_type'].isin(appt_types)) &
    (df['season'].isin(seasons))
]

st.subheader(f"Filtered Appointments ({len(filtered_df)})")
st.dataframe(filtered_df[['age', 'gender', 'lead_time_days', 'no_show_prob', 'predicted_no_show',
                          'intervention', 'appointment_day']])

# Intervention Summary
st.markdown("### 📊 Intervention Summary")
summary = filtered_df['intervention'].value_counts().reset_index()
summary.columns = ['Intervention Type', 'Count']
st.bar_chart(summary.set_index('Intervention Type'))

# CSV Export
csv = filtered_df.to_csv(index=False)
st.download_button("Download Filtered Data as CSV", csv, "filtered_no_show_data.csv")
