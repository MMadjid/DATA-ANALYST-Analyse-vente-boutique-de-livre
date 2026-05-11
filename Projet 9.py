import pandas as pd
import numpy as np
from sklearn.metrics import (silhouette_score)
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


file_path1 = 'datasets/customers.csv'
file_path2 = 'datasets/transactions.csv'
file_path3 = 'datasets/products.csv'
data1 = pd.read_csv(file_path1, sep=",")
data2 = pd.read_csv(file_path2, sep=",")
data3 = pd.read_csv(file_path3, sep=",")

print(data1.head(25))
print(data1.columns)
print(data1.info())
print(data1.isnull().sum())

print(data2.head(25))
print(data2.columns)
print(data2.info())
print(data2.isnull().sum())

print(data3.head(25))
print(data3.columns)
print(data3.info())
print(data3.isnull().sum())

data4=pd.merge(data2, data3, on="id_prod", how="left")
data5=pd.merge(data4, data1, on="client_id", how="left")

data5.to_csv("fichier_final.csv", index=False)

file_path5 = 'fichier_final.csv'
data5 = pd.read_csv(file_path5, sep=",")

print(data5.head(25))
print(data5.columns)
print(data5.info())
print(data5.isnull().sum())

#1) Analyse des ventes
#Q01 : Quel est le chiffre d'affaires global réalisé par la librairie ?

data5 = data5.drop_duplicates()
data5["date"] = pd.to_datetime(data5["date"], errors="coerce") #format date dans excel
data5 = data5.dropna(subset=["date"]) #suprimer erreur dans la date

ca_global = data5["price"].sum()
print("\nQ01 — CHIFFRE D'AFFAIRE")
print(ca_global)

#Q02 : Comment ce chiffre évolue-t-il au fil du temps (mois, trimestres, etc.) ?

ca_mensuel = data5.set_index("date").resample("ME")["price"].sum().reset_index()

print("\nQ02 — EVOLUTION CA AU FIL DU TEMPS ")
print(ca_mensuel)

#Q03 : Quels sont les produits les plus vendus ? Les moins vendus ?
ca_par_produit = (
    data5.groupby("id_prod")["price"]
         .sum()
         .reset_index(name="ca_total"))

print("\nQ03 — CHIFFRE D'AFFAIRE PAR PRODUIT ")
print(ca_par_produit.head())

#Q04 : Quelles sont les catégories de livres les plus rentables ?
ca_par_categorie = data5.groupby("categ")["price"].sum().reset_index()

print("\nQ04 — CATEGORIES LES PLUS RENTABLES")
print(ca_par_categorie)


#2) Analyse des clients

data5 = data5.copy()
data5["date"] = pd.to_datetime(data5["date"], errors="coerce")
data5 = data5.dropna(subset=["date", "price", "client_id", "session_id"])
data5["sex"] = data5["sex"].astype("string").str.lower().str.strip()
data5["age"] = data5["date"].dt.year - data5["birth"]

#Q1 : Quels sont les profils types des clients (âge, genre, fréquence d'achat) ?

profil_clients = (
data5.groupby("client_id")
.agg(
genre=("sex", "first"),
age_moyen=("age", "mean"),
nb_sessions=("session_id", "nunique"),   # fréquence d'achat (visites)
panier_moyen=("price", "mean"),
ca_total=("price", "sum"),
nb_produits_diff=("id_prod", "nunique"),)
.reset_index())

profils_par_genre = (
profil_clients.groupby("genre")
.agg(
nb_clients=("client_id", "nunique"),
age_moyen=("age_moyen", "mean"),
frequence_moy=("nb_sessions", "mean"),
panier_moy=("panier_moyen", "mean"),
ca_moy=("ca_total", "mean"),)
.reset_index())

print("\nQ1 — PROFILS CLIENTS (aperçu)")
print(profil_clients.head(10))
print("\nQ1 — STATISTIQUES GLOBALES")
print(profil_clients[["age_moyen", "nb_sessions", "panier_moyen", "ca_total"]].describe())
print("\nQ1 — PROFILS MOYENS PAR GENRE")
print(profils_par_genre)

#Q2 : Quelle est la relation entre l’âge et le panier moyen ?

corr_age_panier = profil_clients["age_moyen"].corr(profil_clients["panier_moyen"])
age_panier_table = profil_clients[["client_id", "age_moyen", "panier_moyen"]].copy()
bins = [0, 9, 19, 29, 39 , 49, 59, 69, 79, 89, 99]
labels = ["0-9 ans", "10-19 ans", "20-29 ans", "30-39 ans", "40-49 ans", "50-59 ans", "60-69 ans", "70-79 ans", "80-89 ans", "90-99 ans"]
panier_moyen_data = data5.groupby(['session_id','age'])['price'].sum().reset_index(name='panier')
panier_moyen_data['tranche_age'] = pd.cut(panier_moyen_data['age'], bins=bins, labels=labels, right=True)
panier_moyen2 = panier_moyen_data.groupby('tranche_age')['panier'].mean().reset_index(name='panier_moyen2')
print("Le panier moyen est")
print(panier_moyen2)


print("\nQ2 — CORRÉLATION ÂGE / PANIER MOYEN")
print(corr_age_panier)
print("\nQ2 — TABLE ÂGE / PANIER (aperçu)")
print(age_panier_table.head(10))


#Q3 : Les hommes et les femmes achètent-ils les mêmes types de livres ?

genre_categ_ca = (
data5.groupby(["sex", "categ"])["price"]
.sum()
.reset_index(name="ca_total")
.rename(columns={"sex": "genre"}))

genre_categ_volume = (
data5.groupby(["sex", "categ"])["id_prod"]
.count()
.reset_index(name="nb_lignes")
.rename(columns={"sex": "genre"}))

print("\nQ3 — CA PAR GENRE ET CATÉGORIE")
print(genre_categ_ca.head(10))
print("\nQ3 — VOLUME PAR GENRE ET CATÉGORIE")
print(genre_categ_volume.head(10))


#Q4 : Y a-t-il une corrélation entre âge et fréquence d'achat ?
corr_age_freq = profil_clients["age_moyen"].corr(profil_clients["nb_sessions"])

print("\nQ4 — CORRÉLATION ÂGE / FRÉQUENCE (SESSIONS)")
print(corr_age_freq)

#Q5 : Peut-on identifier des groupes de clients selon leur comportement ?


# Variables comportementales
X = profil_clients[["age_moyen", "nb_sessions", "panier_moyen", "ca_total"]].copy()
X = X.replace([np.inf, -np.inf], np.nan).dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto")
clusters = kmeans.fit_predict(X_scaled)

# Associer clusters aux clients
profil_clients.loc[X.index, "cluster"] = clusters

# Profil moyen par cluster
profil_par_cluster = (
profil_clients.groupby("cluster")
.agg(
nb_clients=("client_id", "nunique"),
age_moyen=("age_moyen", "mean"),
frequence_moy=("nb_sessions", "mean"),
panier_moy=("panier_moyen", "mean"),
ca_moy=("ca_total", "mean"),)
.reset_index())

print("\nQ5 — CLIENTS AVEC CLUSTERS (aperçu)")
print(profil_clients[["client_id", "cluster", "age_moyen", "nb_sessions", "panier_moyen", "ca_total"]].head(10))

print("\nQ5 — PROFILS MOYENS PAR CLUSTER")
print(profil_par_cluster)

silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto")
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)
    print(f"K={k} → silhouette score = {score:.3f}")


#fichiers power BI :

out_dir = r"C:\Users\mad77\Desktop\Boulot\DATA\projet 9\powerbi_exports"
import os
os.makedirs(out_dir, exist_ok=True)

ca_mensuel.to_csv(os.path.join(out_dir, "ca_mensuel.csv"), index=False, sep=";", decimal=",")
ca_par_produit.to_csv(os.path.join(out_dir, "ca_par_produit.csv"), index=False, sep=";", decimal=",")
ca_par_categorie.to_csv(os.path.join(out_dir, "ca_par_categorie.csv"), index=False, sep=";", decimal=",")

profil_clients.to_csv(os.path.join(out_dir, "clients_profil.csv"), index=False, sep=";", decimal=",")
profils_par_genre.to_csv(os.path.join(out_dir, "profils_par_genre.csv"), index=False, sep=";", decimal=",")
age_panier_table.to_csv(os.path.join(out_dir, "age_panier.csv"), index=False, sep=";", decimal=",")
genre_categ_ca.to_csv(os.path.join(out_dir, "genre_categ_ca.csv"), index=False, sep=";", decimal=",")
genre_categ_volume.to_csv(os.path.join(out_dir, "genre_categ_volume.csv"), index=False, sep=";", decimal=",")
profil_par_cluster.to_csv(os.path.join(out_dir, "profil_par_cluster.csv"), index=False, sep=";", decimal=",")

corr_table = pd.DataFrame({
    "metric": ["corr_age_panier_moyen", "corr_age_nb_sessions"],
    "value": [corr_age_panier, corr_age_freq]
})
corr_table.to_csv(os.path.join(out_dir, "correlations.csv"), index=False, sep=";", decimal=",")

print("✅ Exports Power BI créés dans :", out_dir)
print(corr_table)

#graphique CA global

plt.figure()
plt.bar(["Chiffre d'affaires global"], [ca_global])
plt.ylabel("Montant (€)")
plt.title("Chiffre d'affaires global de la librairie")

plt.savefig(os.path.join(out_dir, "ca_global.png"), dpi=300, bbox_inches="tight")
plt.close()