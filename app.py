import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="No-Show Predictor", layout="wide")
st.title("📅 Patient No-Show Prediction")

# Load data
df = pd.read_csv("predictions.csv")
df['appointment_day'] = pd.to_datetime(df['appointment_day'])

# Map appointment types
appt_type_map = {
    0: "Consultation",
    1: "Lab Test",
    2: "Follow-up"
}
df['appointment_type'] = df['appointment_type'].map(appt_type_map)

# Map seasons
season_map = {
    0: "Winter",
    1: "Early Spring",
    2: "Late Spring",
    3: "Summer",
    4: "Monsoon",
    5: "Fall",
    6: "Late Fall"
}
df['season'] = df['season'].astype(int).map(season_map)

# Filters
risk_range = st.slider("Filter by No-Show Risk Probability", 0.0, 1.0, (0.5, 1.0), 0.01)

# Always show all appointment types regardless of data content
all_appt_types = list(appt_type_map.values())

appt_types = st.multiselect(
    "Appointment Types",
    options=all_appt_types,
    default=all_appt_types
)

# Always show all seasons regardless of data content
all_seasons = list(season_map.values())

seasons = st.multiselect(
    "Seasons",
    options=all_seasons,
    default=all_seasons
)

date_range = st.date_input("Filter by Appointment Date Range",
                           [df['appointment_day'].min(), df['appointment_day'].max()])

# Filter dataframe with applied filters
filtered_df = df[
    (df['no_show_prob'].between(risk_range[0], risk_range[1])) &
    (df['appointment_type'].isin(appt_types)) &
    (df['season'].isin(seasons)) &
    (df['appointment_day'] >= pd.to_datetime(date_range[0])) &
    (df['appointment_day'] <= pd.to_datetime(date_range[1]))
]

st.subheader(f"Filtered Appointments ({len(filtered_df)})")

# Expandable details for each appointment
for idx, row in filtered_df.iterrows():
    with st.expander(f"Appointment Date: {row['appointment_day'].date()} - Type: {row['appointment_type']}"):
        st.write(f"Age: {row['age']}")
        st.write(f"Gender: {row['gender']}")
        st.write(f"Lead Time (days): {row['lead_time_days']}")
        st.write(f"No-show Probability: {row['no_show_prob']:.2f}")
        st.write(f"Predicted No-Show: {'Yes' if row['predicted_no_show'] == 1 else 'No'}")
        st.write(f"Recommended Intervention: {row['intervention']}")

# Intervention Summary
st.markdown("### 📊 Intervention Summary")
summary = filtered_df['intervention'].value_counts().reset_index()
summary.columns = ['Intervention Type', 'Count']
st.bar_chart(summary.set_index('Intervention Type'))

# Risk Distribution Histogram
fig, ax = plt.subplots()
filtered_df['no_show_prob'].hist(bins=20, ax=ax)
ax.set_title("Distribution of No-Show Risk Probabilities")
ax.set_xlabel("No-Show Probability")
ax.set_ylabel("Number of Appointments")
st.pyplot(fig)

# High-risk alert
high_risk_count = filtered_df[filtered_df['no_show_prob'] > 0.85].shape[0]
if high_risk_count > 0:
    st.warning(f"⚠️ There are {high_risk_count} high-risk appointments! Consider immediate interventions.")

# CSV Export
csv = filtered_df.to_csv(index=False)
st.download_button("Download Filtered Data as CSV", csv, "filtered_no_show_data.csv")
