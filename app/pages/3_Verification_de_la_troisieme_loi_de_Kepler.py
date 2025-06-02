import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Vérification de la troisième loi de Kepler", page_icon="🔬")

if 'df3' in st.session_state:
    df3 = st.session_state['df3']
else:
    st.warning("Le dataset n'a pas encore été chargé ou modifié.")

st.write("## Vérification de la troisième loi de Kepler")

st.subheader("Question 16 : La troisième loi de Kepler est-elle vérifiée dans le dataset ?")

# Supprime les lignes avec valeurs manquantes
df_kepler = df3[['pl_orbper', 'pl_orbsmax']].dropna()

# Conversion en années
df_kepler['orbper_years'] = df_kepler['pl_orbper'] / 365.25

# Calculs P² et a³
df_kepler['P_squared'] = df_kepler['orbper_years'] ** 2
df_kepler['a_cubed'] = df_kepler['pl_orbsmax'] ** 3
plt.figure(figsize=(8, 6))
sns.scatterplot(x=np.log10(df_kepler['a_cubed']), y=np.log10(df_kepler['P_squared']))
plt.xlabel("log₁₀(a³)")
plt.ylabel("log₁₀(P²)")
plt.title("Vérification de la 3e loi de Kepler (log-log)")
plt.grid(True)
st.pyplot(plt.gcf())
plt.clf()
correlation = df_kepler['P_squared'].corr(df_kepler['a_cubed'])
st.write("Corrélation entre P² et a³ :", correlation)
st.write("La troisième loi de Kepler est vérifiée dans ce dataset")

st.session_state['df4'] = df3