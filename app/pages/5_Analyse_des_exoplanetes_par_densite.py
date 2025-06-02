import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Analyse des exoplanètes par densité", page_icon="📡")

if 'df5' in st.session_state:
    df5 = st.session_state['df5']
else:
    st.warning("Le dataset n'a pas encore été chargé ou modifié.")

st.write("## Analyse des exoplanètes par densité")

st.subheader("Question 18 : Partout où c’est possible, calculez la densité de la planète avec la formule suivante, exprimant le rapport entre la masse de la planète et son volume :")

# Conversion des unités (en kg et en m)
MJUP_KG = 1.898e27       # Masse de Jupiter en kg
RJUP_M = 7.1492e7        # Rayon de Jupiter en mètres

# Création d'une copie de df pour éviter conflits
df_density = df5.copy()

# Vérification que les deux colonnes nécessaires ne sont pas nulles
df_density = df_density[(~df_density['pl_bmassj'].isnull()) & (~df_density['pl_radj'].isnull())]

# Conversion des unités
df_density['mass_kg'] = df_density['pl_bmassj'] * MJUP_KG
df_density['radius_m'] = df_density['pl_radj'] * RJUP_M

# Calcul du volume et de la densité
df_density['volume_m3'] = (4/3) * np.pi * df_density['radius_m'] ** 3
df_density['density_kg_m3'] = df_density['mass_kg'] / df_density['volume_m3']

# Conversion de la densité en g/cm³ (optionnel mais courant en astro)
df_density['density_g_cm3'] = df_density['density_kg_m3'] / 1000

# Affichage de quelques résultats
st.write(df_density[['pl_name', 'mass_kg', 'radius_m', 'density_kg_m3', 'density_g_cm3']].head())

st.subheader("Question 19 : Remplissez une nouvelle colonne “composition” avec les labels “rocky” (rocheux) ou “gaseous” (gazeux) suivant la densité de la planète. Pour information, voici les données des planètes du Système Solaire et leurs labels :")

# Ajout d'une colonne composition en fonction de la densité
df_density['composition'] = df_density['density_g_cm3'].apply(
    lambda d: 'rocky' if d >= 3.5 else 'gaseous'
)

# Affichage des résultats
st.write(df_density[['pl_name', 'density_g_cm3', 'composition']].head(10))

st.subheader("Question 20 : Quel critère utiliser pour remplir les labels sur la composition à partir des planètes du système solaire ? Vous pourrez utiliser un critère de seuil et un algorithme de classification supervisée, entraîné sur les données du système solaire, puis comparer les résultats.")

from sklearn.tree import DecisionTreeClassifier, plot_tree

# Données du Système Solaire (exemple simplifié)
# Nom, masse (M_Jup), rayon (R_Jup), densité (g/cm³), composition
solar_data = [
    # rocky planets
    ["Mercure", 0.00017, 0.034, 5.43, "rocky"],
    ["Vénus",   0.00256, 0.087, 5.24, "rocky"],
    ["Terre",   0.00315, 0.089, 5.51, "rocky"],
    ["Mars",    0.00034, 0.047, 3.93, "rocky"],
    # gaseous planets
    ["Jupiter", 1.0,     1.0,   1.33, "gaseous"],
    ["Saturne", 0.299,   0.843, 0.69, "gaseous"],
    ["Uranus",  0.046,   0.357, 1.27, "gaseous"],
    ["Neptune", 0.054,   0.346, 1.64, "gaseous"],
]

# DataFrame des données
solar_df = pd.DataFrame(solar_data, columns=["name", "pl_bmassj", "pl_radj", "density_g_cm3", "composition"])

# Données d'entrainement
X_train = solar_df[["density_g_cm3"]]
y_train = solar_df["composition"]

# Entrainement de l'arbre 
clf = DecisionTreeClassifier(max_depth=None, random_state=None)
clf.fit(X_train, y_train)

# Test de l'arbre sur les explonates

X_test = df_density[["density_g_cm3"]]
df_density["composition_tree_density"] = clf.predict(X_test)

# Affichage de l'arbre
plt.figure(figsize=(10,5))
plot_tree(clf, feature_names=["density"], class_names=clf.classes_, filled=True)
st.pyplot(plt.gcf())
plt.clf()

# Comparaison des deux méthodes
st.dataframe(df_density[["pl_name", "density_g_cm3", "composition", "composition_tree"]].head(10))