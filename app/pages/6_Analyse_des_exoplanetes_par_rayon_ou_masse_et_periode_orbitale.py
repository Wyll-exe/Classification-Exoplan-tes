import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Analyse des exoplanètes par rayon / masse et période orbitale", page_icon="⚗️")

if 'df6' in st.session_state:
    df6 = st.session_state['df6']
else:
    st.warning("Le dataset n'a pas encore été chargé ou modifié.")

st.write("## Analyse des exoplanètes par rayon / masse et période orbitale")

st.subheader("Question 21 : En colorisant vos graphiques à l’aide des colonnes “planet_type” et / ou “composition” (argument hue dans le scatterplot de seaborn), tracer des équivalents des graphiques suivant. Attention à l’échelle utilisée !")

df_density = df6.copy()
# 21.a) Planet Mass [Earth Mass] vs Separation [AU], coloré par planet_type (log–log)
# ---------------------------------------------------------------------------------
# Conversion : pl_bmassj (Jupiter Mass) -> Earth Mass
MJUP_TO_MEARTH = 317.8
df_density["mass_earth"] = df_density["pl_bmassj"] * MJUP_TO_MEARTH

# Séparation en UA : pl_orbsmax
# Préparer le DataFrame pour le plot
plot_df1 = df_density.dropna(subset=["mass_earth", "pl_orbsmax", "planet_type"]).copy()

# Palette personnalisée basée sur planet_type (mêmes labels que dans la Question 17)
palette_planet_type = {
    "mars-sized":             "#FF8C00",
    "earth-sized":            "#2E8B57",
    "super-earth-sized":      "#DC143C",
    "sub-neptune-sized":      "#1E90FF",
    "neptune-sized":          "#4682B4",
    "sub-jupiter-sized":      "#8A2BE2",
    "jupiter-sized":          "#FFD700",
    "super-jupiter-sized":    "#7FFF00"
}

plt.figure(figsize=(10, 7))
ax1 = sns.scatterplot(
    data=plot_df1,
    x="pl_orbsmax",        # Séparation [AU]
    y="mass_earth",        # Masse [Earth Mass]
    hue="planet_type",
    palette=palette_planet_type,
    edgecolor="black",
    alpha=0.75,
    s=45
)
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlim(0.01, 10)
ax1.set_ylim(0.1, 1e4)
ax1.set_title(
    "Planet Mass [M⊕] vs Separation [UA]  (log–log)\nColoré par planet_type",
    fontsize=16, fontweight="bold"
)
ax1.set_xlabel("Séparation [UA] (échelle log)", fontsize=13)
ax1.set_ylabel("Masse [M⊕] (échelle log)", fontsize=13)
legend1 = ax1.legend(
    title="planet_type",
    loc="upper left",
    bbox_to_anchor=(1.02, 1),
    frameon=True,
    fontsize=11,
    title_fontsize=12
)
legend1.get_frame().set_edgecolor("gray")
plt.tight_layout()
st.pyplot(plt.gcf())
plt.clf()

# 21.b) Planet Radius [Earth Radius] vs Orbital Period [jours], coloré par composition (X en log)
# ------------------------------------------------------------------------------------------------
# Conversion : pl_radj (Jupiter Radius) -> Earth Radius
RJUP_TO_REARTH = 11.2
df_density["radius_earth"] = df_density["pl_radj"] * RJUP_TO_REARTH

# Préparer DataFrame pour le plot
plot_df2 = df_density.dropna(subset=["radius_earth", "pl_orbper", "composition"]).copy()

# Palette personnalisée basée sur composition (rocky vs gaseous)
palette_composition = {
    "rocky":  "#A52A2A",
    "gaseous": "#1E90FF"
}

plt.figure(figsize=(10, 7))
ax2 = sns.scatterplot(
    data=plot_df2,
    x="pl_orbper",             # Période orbitale [jours]
    y="radius_earth",          # Rayon [Earth Radius]
    hue="composition",
    palette=palette_composition,
    edgecolor="dimgray",
    alpha=0.80,
    s=50
)
ax2.set_xscale("log")
ax2.set_xlim(0.1, 2000)
ax2.set_ylim(0.5, 22)
ax2.set_title(
    "Planet Radius [R⊕] vs Orbital Period [jours]  (X en log)\nColoré par composition",
    fontsize=16, fontweight="bold"
)
ax2.set_xlabel("Période orbitale [jours] (échelle log)", fontsize=13)
ax2.set_ylabel("Rayon [R⊕]", fontsize=13)
legend2 = ax2.legend(
    title="composition",
    loc="upper right",
    bbox_to_anchor=(1.25, 1),
    frameon=True,
    fontsize=11,
    title_fontsize=12
)
legend2.get_frame().set_edgecolor("gray")
plt.tight_layout()
st.pyplot(plt.gcf())
plt.clf()