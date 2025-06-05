import streamlit as st
import pandas as pd
import os

def file_selector(folder_path='.'):
    # Filtrer pour ne proposer que les fichiers CSV
    filenames = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    selected_filename = st.selectbox('Sélectionnez un fichier CSV', filenames)
    return os.path.join(folder_path, selected_filename)

# Sélection du fichier
folder = '.'  # ou le chemin de ton dossier
filename = file_selector(folder)
st.write('Fichier sélectionné : `%s`' % filename)

# Paramètres pour read_csv
sep = st.text_input("Séparateur", value=",")
skiprows = st.number_input("Nombre de lignes à sauter (skiprows)", min_value=0, value=0, step=1)

# Charger le fichier si l'utilisateur clique sur un bouton
if st.button("Charger le fichier"):
    df = pd.read_csv(filename, sep=sep, skiprows=skiprows)
    st.dataframe(df.head())

st.subheader("Question 2 : Extraire la liste du nom des colonnes du dataset.")
st.dataframe(df.columns)

st.subheader("Question 3 : Quelle est la taille du dataset ?")
st.write("Taille du dataset : (Lignes, colonnes) :")
st.dataframe(df.shape)

st.subheader("Question 4 : Filtrer les colonnes pour ne garder que des valeurs utiles")
list_columns = ["pl_name", "sy_pnum", "pl_orbeccen", "sy_snum", "pl_bmassj", "pl_radj", "pl_orbper", "pl_eqt", "discoverymethod","disc_year", "st_teff", "st_rad", "st_mass", "sy_dist", "pl_orbsmax", "pl_insol", 'pl_controv_flag']
df = df[list_columns]
st.dataframe(df)

st.subheader("Question 5 : Renommer les colonnes pour qu’elles aient des noms plus explicites, si besoin.")
new_names = {
    "st_teff": "stellar_temp",
    "sy_dist": "distance_parsecs",
}
df = df.rename(new_names, axis=1)
st.dataframe(df.head())

st.subheader("Question 6 : Afficher le taux de remplissage de chaque colonne.")
st.write("Taux de remplissage de chaque colonne :")
st.dataframe(df.isnull().mean() * 100)

st.subheader("Question 7 : Repérer où sont les valeurs nulles et proposer une explication. Est-ce que ces valeurs n’existent pas ou est-ce qu’elle n’ont pas été remplies ?")
st.write("Valeurs nulles dans le dataset :")
st.dataframe(df.isnull().sum())
st.write("Ces valeurs nulles existent mais les calculs n'ont pas été effectués pour obtenir les résultats , selon la distance de la planète, sa masse, son rayon, etc et le type de données peut varier.")

st.session_state['df'] = df