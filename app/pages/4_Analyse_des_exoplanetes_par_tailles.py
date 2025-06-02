import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Analyse des exoplanètes par tailles", page_icon="🔭")

if 'df4' in st.session_state:
    df4 = st.session_state['df4']
else:
    st.warning("Le dataset n'a pas encore été chargé ou modifié.")

st.write("## Analyse des exoplanètes par tailles")

st.subheader("Question 17 : Tracer le graphique suivant :")

# 1. Conversion du rayon en rayons terrestres si nécessaire
if 'pl_rade' in df4.columns:
    df4['radius_earth'] = df4['pl_rade']
else:
    df4['radius_earth'] = df4['pl_radj'] * 11.2  # 1 Jupiter radius = 11.2 Earth radii

# 2. Définition des intervalles et labels pour chaque catégorie de planète
bins = [0, 0.53, 1, 1.75, 3.5, 6, 9, 15, 22]
labels = [
    'mars-sized', 
    'earth-sized', 
    'super-earth-sized', 
    'sub-neptune-sized', 
    'neptune-sized', 
    'sub-jupiter-sized', 
    'jupiter-sized', 
    'super-jupiter-sized'
]

# 3. Création de la colonne planet_type
df4['planet_type'] = pd.cut(df4['radius_earth'], bins=bins, labels=labels, right=True, include_lowest=True)

# 4. Comptage des planètes par catégorie
planet_counts = df4['planet_type'].value_counts().reindex(labels, fill_value=0)

# 5. Tracé du barplot
plt.figure(figsize=(12,6))
sns.barplot(x=planet_counts.index, y=planet_counts.values, color='#FFB300')
plt.title("Known transiting planets by size category")
plt.xlabel("Planet type")
plt.ylabel("Number of planets")
plt.tight_layout()
st.pyplot(plt.gcf())
plt.clf()
st.write("Barplot des planètes par type tracé.")

# Affichage de quelques lignes pour contrôle
st.write(df4[['pl_name', 'radius_earth', 'planet_type']].head(10))

st.session_state['df5'] = df4