import streamlit as st

st.set_page_config(page_title="Quelques comptes", page_icon="📈")

if 'df' in st.session_state:
    df = st.session_state['df']
else:
    st.warning("Le dataset n'a pas encore été chargé ou modifié.")

st.write("## Quelques comptes")

st.subheader("Question 8 : Combien y a-t-il de planètes différentes découvertes ?")
st.write("Planètes unique :", df.pl_name.nunique())

st.subheader("Question 9 : Combien y a-t-il de systèmes doubles ? (c’est-à-dire de systèmes de planètes orbitant autour de deux étoiles)")
st.write("Systèmes doubles : ", df[df['sy_snum'] >= 2].shape[0])

st.subheader("Question 10 : Combien y a-t-il de découvertes controversées ?")
st.write("Découvertes controversées : ", df[df['pl_controv_flag'] >= 1].shape[0])

st.subheader("Question 11 : En quelle année a-t-on découvert le plus d’exoplanètes ?")
st.write("Année avec le plus d'exoplanètes découvertes :" , df['disc_year'].mode()[0])

st.subheader("Question 12 : Y a-t-il des doublons ? Vous pourrez utiliser les fonctions duplicated() pour répondre à cette question.	 Vous pouvez explorer la fonction drop_duplicates() de Python ensuite pour vous séparer des doublons.")
st.write("Doublons dans le dataset : ", df["pl_name"].duplicated().sum())
st.write("Nombre de lignes avant suppression des doublons : ", df["pl_name"].shape[0])
st.write("Nombre de lignes après suppression des doublons : ", df["pl_name"].drop_duplicates().shape[0])
colonnes_numeriques = df.select_dtypes(include=['number']).columns
df_num = df[['pl_name'] + list(colonnes_numeriques)]
df_grouped = df_num.groupby('pl_name', as_index=False).mean()
st.write("Nombre de lignes après regroupement par nom de planète : ", df_grouped.shape[0])
st.write(df_grouped.isnull().sum())
st.write(df[["pl_name","pl_orbper"]].head(10))
st.write(df_grouped[["pl_name","pl_orbper"]].head(10))

st.session_state['df2'] = df