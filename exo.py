# 1 - Charger le fichier avec read_csv(). Vous devrez utiliser l’option skip_rows (voir la documentation de pandas) puis appliquer la fonction .head()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
list_columns = ["pl_name", "sy_snum", "pl_bmassj", "pl_radj", "pl_orbper", "pl_eqt", "discoverymethod","disc_year", "st_teff", "st_rad", "st_mass", "sy_dist", "pl_orbsmax", "pl_insol", 'pl_controv_flag']
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


col = 'sy_snum'  # nom de la colonne spécifique

plt.figure(figsize=(8, 4))
sns.histplot(df_grouped[col].dropna(), kde=False) 
plt.title(f'Distribution de {col}')
plt.xlabel(col)
plt.ylabel('Fréquence')
plt.show()
print("Distribution de chaque indicateur tracée.")

# 14 - Tracer un boxplot pour chaque indicateur (fonction boxplot() de seaborn)

# 15 - Calculer la matrice de corrélation et afficher là avec la fonction heatmap() de seaborn. Y a-t-il corrélation entre certaines variables ?

print("\n-----Question 15-----\n")
df_correlation = df_grouped.drop(columns=['pl_name']).corr(numeric_only=True)
print(df_correlation.corr(numeric_only=True))
print(sns.heatmap(df_correlation, annot=True, linewidth=.5))
plt.show()

# Il y a une corrélation entre pl_orbsmax et pl_orbper ainsi qu'une corrélation forte entre pl_insol et pl_eqt

# 16 - La troisième loi de Kepler est-elle vérifiée dans le dataset ?

# 17 - Tracer le graphique suivant :

# 18 - Partout où c’est possible, calculez la densité de la planète avec la formule suivante, exprimant le rapport entre la masse de la planète et son volume :

# 19 - Remplissez une nouvelle colonne “composition” avec les labels “rocky” (rocheux) ou “gaseous” (gazeux) suivant la densité de la planète. Pour information, voici les données des planètes du Système Solaire et leurs labels :

# 20 - Quel critère utiliser pour remplir les labels sur la composition à partir des planètes du système solaire ? Vous pourrez utiliser un critère de seuil et un algorithme de classification supervisée, entraîné sur les données du système solaire, puis comparer les résultats.
