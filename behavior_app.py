import streamlit as st
import numpy as np
import joblib

model = joblib.load('random_forest_model.pkl')

SEUIL_OPTIMAL = 0.5146

st.title("Prédiction de la personnalité : Introverti vs Extraverti")

# Formulaire avec les échelles
time_spent_alone = st.slider("Temps passé seul par jour (heures)", 0, 11, 4)
stage_fear = st.selectbox("Avoir le trac ?", ["Non", "Oui"])
social_event_attendance = st.slider("Fréquence de participations à des événements sociaux (0-10)", 0, 10, 4)
going_outside = st.slider("Fréquence de sorties (0-7)", 0, 7, 3)
drained_after_socializing = st.selectbox("Fatigué après socialisation ?", ["Non", "Oui"])
friends_circle_size = st.slider("Nombre d'amis proches", 0, 15, 5)
post_frequency = st.slider("Fréquence de publication sur réseaux sociaux (0-10)", 0, 10, 3)

if st.button("Prédire la personnalité"):

    stage_fear_bin = 1 if stage_fear == "Oui" else 0
    drained_bin = 1 if drained_after_socializing == "Oui" else 0

    X_new = np.array([[time_spent_alone, social_event_attendance, going_outside,
                       friends_circle_size, post_frequency, stage_fear_bin, drained_bin]])

    proba = model.predict_proba(X_new)[0][1]
    pred = "Introverti" if proba >= SEUIL_OPTIMAL else "Extraverti"

    st.write(f"Probabilité d'être introverti : {proba:.2f}")
    st.success(f"Prédiction finale : {pred}")
