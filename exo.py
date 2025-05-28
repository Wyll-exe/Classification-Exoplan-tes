# 1 - Charger le fichier avec read_csv(). Vous devrez utiliser l’option skip_rows (voir la documentation de pandas) puis appliquer la fonction .head()

import pandas as pd

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
list_columns = ["pl_name", "sy_snum", "pl_bmassj", "pl_radj", "pl_orbper", "pl_eqt", "discoverymethod", "st_teff", "st_rad", "st_mass", "sy_dist", "pl_orbsmax", "pl_insol", 'pl_controv_flag']
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

# 11 - Y a-t-il des doublons ? Vous pourrez utiliser la fonction duplicated() pour répondre à cette question.

# 12 - Tracer la distribution de chaque indicateur (fonction histplot() de seaborn)

# 13 - Tracer un boxplot pour chaque indicateur (fonction boxplot() de seaborn)

# 14 - Calculer la matrice de corrélation et afficher là avec la fonction heatmap() de seaborn. Y a-t-il corrélation entre certaines variables ?

# 15 - La troisième loi de Kepler est-elle vérifiée dans le dataset ?

# 16 - Tracer le graphique suivant :

# 17 - Partout où c’est possible, calculez la densité de la planète avec la formule suivante, exprimant le rapport entre la masse de la planète et son volume.

# 18 - Remplissez une nouvelle colonne “composition” avec les labels “rocky” (rocheux) ou “gaseous” (gazeux) suivant la densité de la planète.

# 19 - Quel critère utiliser pour remplir les labels sur la composition à partir des planètes du système solaire ? Vous pourrez utiliser un critère de seuil et un algorithme de classification supervisée, entraîné sur les données du système solaire, puis comparer les résultats.

# 20 - En colorisant vos graphiques à l’aide des colonnes “planet_type” et / ou “composition” (argument hue dans le scatterplot de seaborn), tracer des équivalents des graphiques suivant. Attention à l’échelle utilisée !