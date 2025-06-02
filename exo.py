# 1 - Charger le fichier avec read_csv(). Vous devrez utiliser l’option skip_rows (voir la documentation de pandas) puis appliquer la fonction .head()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

print("\n-----Question 1-----\n")
m = 96

df = pd.read_csv("dataset_exoplanets.csv",sep=",",skiprows=m)
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
# Boxplot du nombre de planètes par système d'étoiles
sns.boxplot(data=df, x="sy_snum", y="sy_pnum")
plt.title("Nombre de planètes par système d'étoiles")
plt.xlabel("Nombre d'étoiles dans le système")
plt.ylabel("Nombre de planètes")
plt.xticks(rotation=45)
plt.show()
print("Boxplot pour chaque indicateur tracé.")


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
plt.show()

# Comparaison des deux méthodes
print(df_density[["pl_name", "density_g_cm3", "composition", "composition_tree"]].head(10))