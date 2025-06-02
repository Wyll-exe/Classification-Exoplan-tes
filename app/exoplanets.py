import streamlit as st
import pandas as pd

st.set_page_config(page_title="Exoplanètes", page_icon="🪐")

st.title("Projet exoplanètes")
st.write("## Premiers traitements / analyses à mener")

m = 96

df = pd.read_csv("../dataset_exoplanets.csv",sep=",",skiprows=m)

st.subheader("Question 1 : Charger le fichier avec read_csv(). Vous devrez utiliser l’option skip_rows (voir la documentation de pandas) puis appliquer la fonction .head()")
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