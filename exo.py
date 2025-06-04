# 1 - Charger le fichier avec read_csv(). Vous devrez utiliser l’option skip_rows (voir la documentation de pandas) puis appliquer la fonction .head()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

print("\n-----Question 1-----\n")
m = 96

df = pd.read_csv("app/dataset_exoplanets.csv",sep=",",skiprows=m)
print(df.head())

# 2 - Extraire la liste du nom des colonnes du dataset.

print("\n-----Question 2-----\n")
print(df.columns)

# 3 - Quelle est la taille du dataset ?

print("\n-----Question 3-----\n")
print("Taille du dataset : (Lignes, colonnes) :", df.shape)

# 4 - Filtrer les colonnes pour ne garder que des valeurs utiles

print("\n-----Question 4-----\n")
list_columns = ["pl_name", "sy_pnum", "pl_orbeccen", "sy_snum", "pl_bmassj", "pl_radj", "pl_orbper", "pl_eqt", "discoverymethod","disc_year", "st_teff", "st_rad", "st_mass", "sy_dist", "pl_orbsmax", "pl_insol", 'pl_controv_flag']
df = df[list_columns]
print(df)

# 5 - Renommer les colonnes pour qu’elles aient des noms plus explicites, si besoin.

print("\n-----Question 5-----\n")
new_names = {
    "st_teff": "stellar_temp",
    "sy_dist": "distance_parsecs",
}
df = df.rename(new_names, axis=1)
print(df)

# 6 - Afficher le taux de remplissage de chaque colonne.

print("\n-----Question 6-----\n")
print("Taux de remplissage de chaque colonne :")
print(df.isnull().mean() * 100)

# 7 - Repérer où sont les valeurs nulles et proposer une explication. Est-ce que ces valeurs n’existent pas ou est-ce qu’elle n’ont pas été remplies ?

print("\n-----Question 7-----\n")
print("Valeurs nulles dans le dataset :")
print(df.isnull().sum())

# ces valeurs nulles existent mais les calculs n'ont pas été effectués pour obtenir les résultats , selon la distance de la planète, sa masse, son rayon, etc et le type de données peut varier.


# 8 - Combien y a-t-il de planètes différentes découvertes ?

print("\n-----Question 8-----\n")
print("Planètes unique : ", df.pl_name.nunique())

# 9 - Combien y a-t-il de systèmes doubles ? (c’est-à-dire de systèmes de planètes orbitant autour de deux étoiles)

print("\n-----Question 9-----\n")
print("Systèmes doubles : ", df[df['sy_snum'] >= 2].shape[0])

# 10 - Combien y a-t-il de découvertes controversées ?

print("\n-----Question 10-----\n")
print("Découvertes controversées : ", df[df['pl_controv_flag'] >= 1].shape[0])

# 11 - En quelle année a-t-on découvert le plus d’exoplanètes ?

print("\n-----Question 11-----\n")
print("Année avec le plus d'exoplanètes découvertes :" , df['disc_year'].mode()[0])

# 12 - Y a-t-il des doublons ? Vous pourrez utiliser les fonctions duplicated() pour répondre à cette question.	 Vous pouvez explorer la fonction drop_duplicates() de Python ensuite pour vous séparer des doublons.

print("\n-----Question 12-----\n")
print("Doublons dans le dataset : ", df["pl_name"].duplicated().sum())
print("Nombre de lignes avant suppression des doublons : ", df["pl_name"].shape[0])
print("Nombre de lignes après suppression des doublons : ", df["pl_name"].drop_duplicates().shape[0])
colonnes_numeriques = df.select_dtypes(include=['number']).columns
df_num = df[['pl_name'] + list(colonnes_numeriques)]
df_grouped = df_num.groupby('pl_name', as_index=False).mean()
print("Nombre de lignes après regroupement par nom de planète : ", df_grouped.shape[0])
print(df_grouped.isnull().sum())

print(df[["pl_name","pl_orbper"]].head(10))
print(df_grouped[["pl_name","pl_orbper"]].head(10))
# 13 - Tracer la distribution de chaque indicateur (fonction histplot() de seaborn)

print("\n-----Question 13-----\n")

print("exemple 1")

col = 'sy_snum'  # nom de la colonne spécifique

plt.figure(figsize=(8, 4))
sns.histplot(df_grouped[col].dropna(), kde=False) 
plt.title(f'Distribution de {col}')
plt.xlabel(col)
plt.ylabel('Fréquence')
plt.show()
print("Distribution de chaque indicateur tracée.")

print("exemple 2")

col = 'sy_pnum'  # nom de la colonne spécifique

plt.figure(figsize=(8, 4))
sns.histplot(df_grouped[col].dropna(), kde=False) 
plt.title(f'Distribution de {col}')
plt.xlabel(col)
plt.ylabel('Fréquence')
plt.show()
print("Distribution de chaque indicateur tracée.")

print("exemple 3")

col = 'pl_orbeccen'  # nom de la colonne spécifique

plt.figure(figsize=(8, 4))
sns.histplot(df_grouped[col].dropna(), kde=False) 
plt.title(f'Distribution de {col}')
plt.xlabel(col)
plt.ylabel('Fréquence')
plt.show()
print("Distribution de chaque indicateur tracée.")


# 14 - Tracer un boxplot pour chaque indicateur (fonction boxplot() de seaborn)

print("\n-----Question 14-----\n")
plt.figure(figsize=(14, 6))
sns.boxplot(data=df, x="disc_year", y="pl_bmassj")
plt.title("Distribution de la masse des planètes par année de découverte")
plt.xlabel("Année de découverte")
plt.ylabel("Masse des planètes (pl_bmassj)")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(14, 6))
sns.boxplot(data=df, x="disc_year", y="sy_snum")
plt.title("Distribution du nombre de systèmes par année de découverte")
plt.xlabel("Année de découverte")
plt.ylabel("Nombre de systèmes (sy_snum)")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(8, 4))
sns.boxplot(y=df["pl_radj"])
plt.title("Distribution du rayon des planètes")
plt.ylabel("Rayon des planètes (pl_radj)")
plt.show()


plt.figure(figsize=(8, 4))
sns.boxplot(y=df["pl_eqt"])
plt.title("Distribution de la température d'équilibre des planètes")
plt.ylabel("Température d'équilibre (pl_eqt)")
plt.show()


# 15 - Calculer la matrice de corrélation et afficher là avec la fonction heatmap() de seaborn. Y a-t-il corrélation entre certaines variables ?

print("\n-----Question 15-----\n")
df_correlation = df_grouped.drop(columns=['pl_name']).corr(numeric_only=True)
print(df_correlation.corr(numeric_only=True))
print(sns.heatmap(df_correlation, annot=True, linewidth=.5))
plt.show()

# Il y a une corrélation entre pl_orbsmax et pl_orbper ainsi qu'une corrélation forte entre pl_insol et pl_eqt

# 16 - La troisième loi de Kepler est-elle vérifiée dans le dataset ?

print("\n-----Question 16-----\n")
# Supprime les lignes avec valeurs manquantes
df_kepler = df[['pl_orbper', 'pl_orbsmax']].dropna()

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
plt.show()
correlation = df_kepler['P_squared'].corr(df_kepler['a_cubed'])
print("Corrélation entre P² et a³ :", correlation)

# La troisième loi de Kepler est vérifiée dans ce dataset

# 17 - Tracer le graphique suivant :

print("\n-----Question 17-----\n")

# 1. Conversion du rayon en rayons terrestres si nécessaire
if 'pl_rade' in df.columns:
    df['radius_earth'] = df['pl_rade']
else:
    df['radius_earth'] = df['pl_radj'] * 11.2  # 1 Jupiter radius = 11.2 Earth radii

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
df['planet_type'] = pd.cut(df['radius_earth'], bins=bins, labels=labels, right=True, include_lowest=True)

# 4. Comptage des planètes par catégorie
planet_counts = df['planet_type'].value_counts().reindex(labels, fill_value=0)

# 5. Tracé du barplot
plt.figure(figsize=(12,6))
sns.barplot(x=planet_counts.index, y=planet_counts.values, color='#FFB300')
plt.title("Known transiting planets by size category")
plt.xlabel("Planet type")
plt.ylabel("Number of planets")
plt.tight_layout()
plt.show()
print("Barplot des planètes par type tracé.")

# Affichage de quelques lignes pour contrôle
print(df[['pl_name', 'radius_earth', 'planet_type']].head(10))



# 18 - Partout où c’est possible, calculez la densité de la planète avec la formule suivante, exprimant le rapport entre la masse de la planète et son volume :

print("\n-----Question 18-----\n")
# Conversion des unités (en kg et en m)
MJUP_KG = 1.898e27       # Masse de Jupiter en kg
RJUP_M = 7.1492e7        # Rayon de Jupiter en mètres

# Création d'une copie de df pour éviter conflits
df_density = df.copy()

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
print(df_density[['pl_name', 'mass_kg', 'radius_m', 'density_kg_m3', 'density_g_cm3']].head())

# 19 - Remplissez une nouvelle colonne “composition” avec les labels “rocky” (rocheux) ou “gaseous” (gazeux) suivant la densité de la planète. Pour information, voici les données des planètes du Système Solaire et leurs labels :

print("\n-----Question 19-----\n")
# Ajout d'une colonne composition en fonction de la densité
df_density['composition'] = df_density['density_g_cm3'].apply(
    lambda d: 'rocky' if d >= 3.5 else 'gaseous'
)

# Affichage des résultats
print(df_density[['pl_name', 'density_g_cm3', 'composition']].head(10))

# 20 - Quel critère utiliser pour remplir les labels sur la composition à partir des planètes du système solaire ? Vous pourrez utiliser un critère de seuil et un algorithme de classification supervisée, entraîné sur les données du système solaire, puis comparer les résultats.
from sklearn.tree import DecisionTreeClassifier, plot_tree

print("\n-----Question 20-----\n")
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
df_density["composition_tree"] = clf.predict(X_test)

# Affichage de l'arbre
plt.figure(figsize=(10,5))
plot_tree(clf, feature_names=["density"], class_names=clf.classes_, filled=True)
plt.show()

# Comparaison des deux méthodes
print(df_density[["pl_name", "density_g_cm3", "composition", "composition_tree"]].head(10))



# 21 - En colorisant vos graphiques à l’aide des colonnes “planet_type” et / ou “composition” (argument hue dans le scatterplot de seaborn), tracer des équivalents des graphiques suivant. Attention à l’échelle utilisée !

print("\n-----Question 21 (Bonus) : scatterplots colorés-----\n")

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
plt.show()

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
plt.show()

print("Plots 21.a et 21.b générés avec succès.")



