import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Analyse univariée et bivariée", page_icon="🧐")

if 'df2' in st.session_state:
    df2 = st.session_state['df2']
else:
    st.warning("Le dataset n'a pas encore été chargé ou modifié.")

st.write("## Analyse univariée et bivariée")

st.subheader("Question 13 : Tracer la distribution de chaque indicateur (fonction histplot() de seaborn)")

colonnes_numeriques = df2.select_dtypes(include=['number']).columns
df_num = df2[['pl_name'] + list(colonnes_numeriques)]
df_grouped = df_num.groupby('pl_name', as_index=False).mean()

st.write("exemple 1")

col = 'sy_snum'  # nom de la colonne spécifique

plt.figure(figsize=(8, 4))
sns.histplot(df_grouped[col].dropna(), kde=False) 
plt.title(f'Distribution de {col}')
plt.xlabel(col)
plt.ylabel('Fréquence')
st.pyplot(plt.gcf())
plt.clf()
st.write("Distribution de chaque indicateur tracée.")

st.write("exemple 2")

col = 'sy_pnum'  # nom de la colonne spécifique

plt.figure(figsize=(8, 4))
sns.histplot(df_grouped[col].dropna(), kde=False) 
plt.title(f'Distribution de {col}')
plt.xlabel(col)
plt.ylabel('Fréquence')
st.pyplot(plt.gcf())
plt.clf()
st.write("Distribution de chaque indicateur tracée.")

st.write("exemple 3")

col = 'pl_orbeccen'  # nom de la colonne spécifique

plt.figure(figsize=(8, 4))
sns.histplot(df_grouped[col].dropna(), kde=False) 
plt.title(f'Distribution de {col}')
plt.xlabel(col)
plt.ylabel('Fréquence')
st.pyplot(plt.gcf())
plt.clf()
st.write("Distribution de chaque indicateur tracée.")

st.subheader("Question 14 : Tracer un boxplot pour chaque indicateur (fonction boxplot() de seaborn)")

# Boxplot du nombre de planètes par système d'étoiles
sns.boxplot(data=df2, x="sy_snum", y="sy_pnum")
plt.title("Nombre de planètes par système d'étoiles")
plt.xlabel("Nombre d'étoiles dans le système")
plt.ylabel("Nombre de planètes")
plt.xticks(rotation=45)
st.pyplot(plt.gcf())
plt.clf()
st.write("Boxplot pour chaque indicateur tracé.")

st.subheader("Question 15 : Calculer la matrice de corrélation et afficher là avec la fonction heatmap() de seaborn. Y a-t-il corrélation entre certaines variables ?")
df_correlation = df_grouped.drop(columns=['pl_name']).corr(numeric_only=True)
st.dataframe(df_correlation.corr(numeric_only=True))
fig, ax = plt.subplots()
sns.heatmap(df_correlation, annot=True, linewidth=.5, ax=ax)
st.pyplot(fig)
plt.clf()
st.write("Il y a une corrélation entre pl_orbsmax et pl_orbper ainsi qu'une corrélation forte entre pl_insol et pl_eqt")

st.session_state['df3'] = df2