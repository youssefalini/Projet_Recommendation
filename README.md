# 🎮 TEKLEVELUP - AI Recommendation Engine

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://projet-recommendation.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![NLP](https://img.shields.io/badge/NLP-Sentence--Transformers-green.svg)
![Machine Learning](https://img.shields.io/badge/Machine_Learning-Scikit--Learn-orange.svg)

## 🚀 Démo en direct
**Testez l'application immédiatement ici :** [https://projet-recommendation.streamlit.app/](https://projet-recommendation.streamlit.app/)

---

## 📌 À propos du projet
**TEKLEVELUP Recommender** est un système de recommandation de matériel e-sport et gaming de niveau production. Contrairement aux algorithmes classiques basés uniquement sur des mots-clés, ce projet implémente un **moteur hybride** capable de comprendre le sens profond des requêtes (Sémantique), d'analyser le comportement des utilisateurs (Collaboratif) et d'optimiser les ventes (Intelligence Business).

## ✨ Fonctionnalités Principales

### 🧠 1. Moteur Hybride de Traitement du Langage Naturel (NLP)
* **Cerveau Lexical (TF-IDF) :** Recherche de précision basée sur les mots-clés exacts.
* **Cerveau Sémantique (Deep Learning) :** Utilisation du modèle `paraphrase-multilingual-MiniLM-L12-v2` pour créer des *embeddings* et comprendre le contexte des descriptions produits.
* **Moteur de Recherche Magique :** Permet à l'utilisateur de taper des phrases naturelles (ex: *"Je cherche un clavier silencieux pour streamer la nuit"*).

### 💼 2. Intelligence Business & Filtrage Collaboratif
* **Analyse Comportementale :** Recommande des produits fréquemment achetés ensemble (Cross-selling) en analysant l'historique des paniers clients.
* **Règles Métier E-commerce :** L'algorithme ajuste dynamiquement les scores en temps réel :
  * 🔴 **Rupture de stock :** Exclusion immédiate des recommandations.
  * 💰 **Marges élevées :** Bonus algorithmique (+20%) pour maximiser la rentabilité.
  * 📉 **Faibles marges :** Malus léger (-10%).
* **Système de Feedback :** Boutons de vote (Like/Dislike) influençant les futures recommandations (+10% par vote positif).

### 📊 3. Data Visualisation Avancée
* **Knowledge Graph (Galaxie IA) :** Cartographie 2D interactive générée avec `NetworkX` et `PyVis` illustrant la façon dont l'IA connecte les produits entre eux (clustering naturel).
* **Radar de Compatibilité (Spider Chart) :** Graphiques `Plotly` expliquant à l'utilisateur la composition exacte du score de recommandation (Sémantique IA vs Communauté vs Avis).

### 🎨 4. UI/UX "Cyberpunk E-Sport"
* Interface Web full-stack propulsée par **Streamlit**.
* Injection de CSS personnalisé (Mode Kiosk, Police Orbitron, Effets Néon/Glow).
* Animations vectorielles dynamiques avec **Lottie**.

---

## 🛠️ Architecture Technique (Stack)

* **Langage :** Python
* **Interface Web :** Streamlit, Streamlit-Lottie
* **Machine Learning / NLP :** Scikit-Learn, Sentence-Transformers
* **Data Manipulation :** Pandas, NumPy
* **Data Visualisation :** Plotly Express, NetworkX, PyVis

---

## 🚀 Installation et Lancement Local

1. **Cloner le dépôt :**
   ```bash
   git clone [https://github.com/youssefalini/Projet_Recommendation.git](https://github.com/youssefalini/Projet_Recommendation.git)
   cd Projet_Recommendation
