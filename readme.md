📈 Sales Forecasting – Data Science Project (M5)
🎯 Objectif du projet

Ce projet vise à prédire les ventes à J+7 à partir de données historiques et calendaires, dans un contexte retail inspiré du challenge M5 Forecasting.
L’objectif est double :

démontrer une démarche Data Scientist complète (de l’analyse à la mise en production),

proposer une solution simple, interprétable et déployable.

🧠 Contexte & données

Dataset : M5 Forecasting (Kaggle)

Périmètre : ventes quotidiennes agrégées pour l’État de Californie

Horizon de prédiction : 7 jours

Variables utilisées :

calendrier (jour de semaine, mois, année, week-end, événements),

historique des ventes (lag J-7, moyenne glissante sur 7 jours).

🛠️ Méthodologie
1. Analyse exploratoire (EDA)

Analyse de la série temporelle des ventes

Mise en évidence de la saisonnalité hebdomadaire

Étude des corrélations entre variables explicatives et cible

⚙️ Comment marche la prédiction
Entrées du modèle

Le modèle reçoit :

des variables calendaires (wday, month, year, is_weekend, is_event)

des variables d’historique des ventes (sales_lag_7, sales_rolling_mean_7)

Ces variables sont connues au moment de la prédiction, ce qui évite toute fuite de données.

Modèle

Régression linéaire avec standardisation

Choisie car :

plus performante que le naïf,

plus simple et interprétable que Random Forest,

cohérente avec le signal observé dans les données.

Sortie

Une valeur numérique : prévision des ventes à J+7.

🚀 Comment le projet est utilisé
1. API FastAPI

Endpoint /predict : reçoit les variables → renvoie la prédiction

Validation métier intégrée (bornes sur les dates, ventes ≥ 0, variables binaires)

Chaque prédiction est loggée (inputs, output, latence)

Endpoints supplémentaires :

/health : état de l’API

/metrics : monitoring simple

👉 Usage typique : intégration dans un outil de pilotage ou un autre service.

2. Application Streamlit

Interface simple pour tester le modèle sans code

L’utilisateur renseigne les variables → obtient la prévision

Déployée sur Streamlit Community Cloud

Redéploiement automatique à chaque push GitHub

👉 Usage typique : démo portfolio / métier.

🧪 Qualité et fiabilité

Tests pytest pour vérifier le bon fonctionnement de l’API

CI GitHub Actions :

installe l’environnement,

lance les tests automatiquement,

garantit que le projet est reproductible.


🔁 Comment reproduire le projet
1. Cloner le dépôt
git clone <repo-url>
cd <repo>

2. Installer les dépendances
pip install -r requirements.txt

3. Lancer l’API
uvicorn api.main:app --reload

4. Lancer l’application Streamlit
streamlit run streamlit_app.py

5. Lancer les tests
pytest



⚠️ Hypothèses et limites

Le modèle est valable uniquement dans le périmètre 2011–2016.

Toute utilisation sur des années plus récentes nécessite un ré-entraînement.

Le projet vise un équilibre entre performance, simplicité et déployabilité, pas l’optimisation extrême.