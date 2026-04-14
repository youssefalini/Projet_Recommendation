import streamlit as st
import numpy as np
import plotly.express as px  # Pour générer le Radar de compatibilité (Spider Chart)
import pandas as pd

# --- Librairies d'Intelligence Artificielle ---
from sentence_transformers import SentenceTransformer  # Modèle Deep Learning (Compréhension sémantique / Sens)
from sklearn.feature_extraction.text import TfidfVectorizer  # Analyse lexicale : Mots exacts (Term Frequency)
from sklearn.metrics.pairwise import \
    cosine_similarity  # Algorithme mathématique pour calculer la distance entre 2 vecteurs

# --- Librairies de Data Visualisation (Graphes) ---
import networkx as nx  # Création et manipulation des nœuds et des liens du graphe
from pyvis.network import Network  # Visualisation interactive du graphe dans un navigateur
import streamlit.components.v1 as components  # Permet d'intégrer du code HTML/JS directement dans Streamlit

# --- Librairies pour les animations ---
import requests  # Pour interroger les serveurs Lottie
from streamlit_lottie import st_lottie  # Pour afficher les animations vectorielles fluides


# ==========================================
# FONCTION UTILITAIRE : CHARGEMENT DES ANIMATIONS
# ==========================================
@st.cache_data
def load_lottieurl(url: str):
    """Télécharge l'animation JSON depuis l'URL fournie de manière sécurisée."""
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


# ==========================================
# 0. INJECTION DU THÈME CYBERPUNK (CSS)
# ==========================================
# Ce bloc personnalise l'interface de base de Streamlit pour lui donner un look E-Sport
st.markdown("""
<style>
    /* Import de la police futuriste Orbitron depuis Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');

    /* Application de la police Orbitron aux titres et textes importants */
    h1, h2, h3, .stMarkdown p strong {
        font-family: 'Orbitron', sans-serif !important;
        letter-spacing: 1px;
    }

    /* Effet Néon Vert sur tous les boutons (Survol = Effet Glow) */
    .stButton>button {
        border: 2px solid #00ff00 !important;
        color: #00ff00 !important;
        background-color: transparent !important;
        border-radius: 8px !important;
        box-shadow: 0 0 10px rgba(0, 255, 0, 0.2) !important;
        transition: all 0.3s ease-in-out !important;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #00ff00 !important;
        color: #0E1117 !important;
        box-shadow: 0 0 20px rgba(0, 255, 0, 0.6) !important;
        transform: scale(1.05);
    }

    /* Stylisation des Cartes Produits (Fond vert très transparent et ombres) */
    div[data-testid="stAlert"] {
        background-color: rgba(0, 255, 0, 0.05);
        border: 1px solid rgba(0, 255, 0, 0.3);
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.5);
    }

    /* MODE KIOSK : Cache le menu hamburger, le header et le footer de base de Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. CONFIGURATION DE LA PAGE
# ==========================================
st.set_page_config(page_title="TEK LEVELUP Recommender", page_icon="🎮", layout="wide")
st.title("🎮 Moteur de Recommandation - TEK LEVELUP")
st.markdown("Découvrez comment l'IA analyse les fiches produits pour suggérer le meilleur setup gaming.")


# ==========================================
# 2. CHARGEMENT DE LA BASE DE DONNÉES (CATALOGUE)
# ==========================================
@st.cache_data
def load_data():
    df = pd.read_csv('catalogue_teklevelup.csv')

    # PRÉPARATION NLP : On fusionne toutes les infos textuelles en une seule "super colonne"
    # C'est ce texte complet que l'IA va lire et analyser.
    df['super_description'] = df['produit'] + " " + df['categorie'] + " " + df['description']

    # --- SIMULATION INTELLIGENCE BUSINESS ---
    np.random.seed(42)  # Le seed garantit que les données aléatoires restent les mêmes à chaque rechargement

    # 1. Simulation des stocks (0 à 20 unités)
    df['stock'] = np.random.randint(0, 20, size=len(df))
    df.loc[0, 'stock'] = 0  # Produit forcé en rupture pour tester l'algorithme de pénalité
    df.loc[3, 'stock'] = 0

    # 2. Simulation des marges bénéficiaires (Faible, Moyenne, Élevée)
    df['marge'] = np.random.choice(['Basse', 'Moyenne', 'Élevée'], size=len(df), p=[0.3, 0.5, 0.2])

    return df


df = load_data()


# ==========================================
# 3. FONCTIONS HISTORIQUES ET FEEDBACKS
# ==========================================

def get_collaborative_scores(panier_ids, total_produits):
    """
    FILTRAGE COLLABORATIF : Analyse ce que les autres clients ont acheté avec ces produits.
    Cette fonction retourne un score "Social" basé sur le comportement réel, pas sur le texte.
    """
    try:
        df_hist = pd.read_csv('historique_achats.csv')
    except FileNotFoundError:
        return np.zeros(total_produits)  # Si pas d'historique, tous les scores sont à zéro

    co_scores = np.zeros(total_produits)

    for p_id in panier_ids:
        # Étape 1 : Trouver tous les utilisateurs qui ont acheté le produit ciblé
        users = df_hist[df_hist['produit_id'] == (p_id + 1)]['user_id'].unique()
        # Étape 2 : Regarder l'historique complet de ces utilisateurs
        autres_achats = df_hist[df_hist['user_id'].isin(users)]['produit_id'].values

        # Étape 3 : Attribuer +1 point de recommandation à ces produits co-achetés
        for a_id in autres_achats:
            if a_id - 1 < total_produits:
                co_scores[a_id - 1] += 1

                # Étape 4 : Normalisation (Ramener le meilleur score à 1.0 pour l'intégrer aux autres algos)
    if co_scores.max() > 0:
        co_scores = co_scores / co_scores.max()
    return co_scores


def enregistrer_feedback(produit_id, action):
    """Enregistre un vote Utilisateur (+1 pour Like, -1 pour Dislike) dans un fichier CSV."""
    try:
        df_feed = pd.read_csv('feedbacks.csv')
    except FileNotFoundError:
        df_feed = pd.DataFrame(columns=['produit_id', 'vote'])

    nouveau_vote = 1 if action == "like" else -1
    nouvelle_ligne = pd.DataFrame({'produit_id': [produit_id], 'vote': [nouveau_vote]})
    df_feed = pd.concat([df_feed, nouvelle_ligne], ignore_index=True)
    df_feed.to_csv('feedbacks.csv', index=False)


# ==========================================
# 4. INITIALISATION DES CERVEAUX I.A. (NLP)
# ==========================================

# --- CERVEAU 1 : Le Modèle Lexical (TF-IDF) ---
# Objectif : Compter les mots-clés exacts en ignorant les mots de liaison (stop words).
mots_inutiles = ['le', 'la', 'les', 'un', 'une', 'des', 'avec', 'pour', 'et', 'en', 'de', 'du', 'au', 'aux', 'qui',
                 'que', 'très']

# Configuration Expert : On lit les mots simples (1) et les paires de mots (2)
vectorizer = TfidfVectorizer(stop_words=mots_inutiles, ngram_range=(1, 2))
# Transformation du texte complet en une matrice de nombres (Vecteurs)
matrice_mots = vectorizer.fit_transform(df['super_description'])


# --- CERVEAU 2 : Le Modèle Sémantique (Deep Learning) ---
# Objectif : Comprendre le sens global de la phrase, même si les mots sont différents.
@st.cache_resource  # Mise en cache pour éviter de re-télécharger le modèle à chaque interaction
def load_ai_model():
    # Modèle 'MiniLM' : Rapide, multi-langues (comprend le français) et parfait pour les phrases courtes.
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')


modele_ia = load_ai_model()

# Transformation des textes en 'Embeddings' (des vecteurs de sens denses)
matrice_ia = np.asarray(modele_ia.encode(df['super_description'].tolist()))

# ==========================================
# 5. MOTEUR DE RECHERCHE MAGIQUE (HYBRIDE)
# ==========================================
st.subheader("✨ Recherche Magique (Langage Naturel)")
st.info("💡 Testez l'IA : Tapez une phrase complète au lieu d'un simple mot-clé.")

recherche_texte = st.text_input(
    "Ex: 'Je cherche un clavier qui ne fait pas de bruit la nuit' ou 'Matériel pour streamer' :")

if recherche_texte:
    st.markdown("### 🎯 L'IA a trouvé ceci pour vous (Recherche Hybride) :")

    # 1. Analyse Sémantique de la requête de l'utilisateur
    vecteur_recherche_ia = np.asarray(modele_ia.encode([recherche_texte])).reshape(1, -1)
    scores_ia = cosine_similarity(vecteur_recherche_ia, matrice_ia).flatten()

    # 2. Analyse Lexicale de la requête (mots exacts)
    vecteur_recherche_mots = vectorizer.transform([recherche_texte])
    scores_mots = cosine_similarity(vecteur_recherche_mots, matrice_mots).flatten()

    # 3. FUSION HYBRIDE : On donne 70% d'importance aux mots exacts (pour cibler le bon type de produit)
    # et 30% au sens (pour comprendre le contexte : "silencieux", "gamer", etc.)
    scores_recherche = (scores_ia * 0.3) + (scores_mots * 0.7)

    # Récupération des 3 meilleurs scores
    indices_top_recherche = scores_recherche.argsort()[::-1][:3]

    cols_rech = st.columns(3)
    for i, idx_rech in enumerate(indices_top_recherche):
        prod_rech = df.iloc[idx_rech]
        score_rech = scores_recherche[idx_rech]

        with cols_rech[i]:
            st.success(f"**{prod_rech['produit']}**")
            st.caption(f"Catégorie : {prod_rech['categorie']} | ⭐️ Note: {prod_rech['note_sur_5']}/5")
            st.write(prod_rech['description'])
            st.progress(min(float(score_rech) * 1.5, 1.0), text="Pertinence Hybride")

st.divider()

# ==========================================
# 6. PANIER INTELLIGENT ET ALGORITHME DE RECOMMANDATION
# ==========================================
st.subheader("🛒 Votre Panier")

panier = st.multiselect(
    "Ajoutez des articles à votre panier pour personnaliser la recommandation :",
    options=df['produit'].tolist())

# GESTION DU PANIER VIDE (Affichage Animation)
if not panier:
    st.warning("Ajoutez au moins un article pour voir les recommandations.")

    lottie_empty = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_qh5z2fdq.json")
    if lottie_empty:
        st_lottie(lottie_empty, height=250, key="empty")

    with st.sidebar:
        st.header("⚙️ Espace Admin")
        st.info("Ajoutez des produits pour voir les métriques.")

else:
    # 1. Récupération des indices des produits dans le panier
    indices_panier = df[df['produit'].isin(panier)].index.tolist()

    # 2. Création du "Sac de Mots" du panier : on rassemble le vocabulaire pour l'explicabilité
    mots_panier = set()
    for idx in indices_panier:
        mots_article = set(df.iloc[idx]['super_description'].lower().split())
        mots_panier = mots_panier.union(mots_article)

        # 3. Chargement des feedbacks de la communauté (Likes / Dislikes)
    try:
        df_feed = pd.read_csv('feedbacks.csv')
        scores_feedback_global = df_feed.groupby('produit_id')['vote'].sum().to_dict()
    except:
        scores_feedback_global = {}

    # 4. CALCULS MATHÉMATIQUES DES SCORES
    # Vecteur moyen : L'IA crée un "produit idéal" qui serait le mélange mathématique de tout le panier
    vecteur_panier = np.asarray(matrice_ia[indices_panier].mean(axis=0)).reshape(1, -1)
    # Score Sémantique IA
    scores_sim = cosine_similarity(vecteur_panier, matrice_ia).flatten()
    # Score Social Historique
    scores_hist = get_collaborative_scores(indices_panier, len(df))

    # 5. DÉCLENCHEUR DU MODE BUSINESS
    st.info("💼 Mode Business : L'IA prend en compte l'argent (Marges) et la réalité (Stocks).")
    col_bouton, col_anim = st.columns([3, 1])

    with col_bouton:
        activer_business = st.checkbox("Activer l'Intelligence Business", key="bouton_ia_business")

    with col_anim:
        if activer_business:
            # Affichage de l'IA au travail (Animation Lottie)
            lottie_ai = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_V9t630.json")
            if lottie_ai:
                st_lottie(lottie_ai, height=80, key="ai_brain")

    # 6. APPLICATION DES PONDÉRATIONS ET RÈGLES
    recommandations_calculees = []

    for idx in range(len(df)):
        # Règle d'or : On ne recommande jamais un produit qui est DÉJÀ dans le panier
        if idx not in indices_panier:
            score_texte = scores_sim[idx]
            note_normalisee = df.iloc[idx]['note_sur_5'] / 5.0
            score_social = scores_hist[idx]

            # --- GESTION DU COLD START ---
            if score_social == 0.0:
                # Produit récent ou jamais acheté : On se base uniquement sur la pertinence du texte (80%) et sa note (20%)
                score_final = (score_texte * 0.8) + (note_normalisee * 0.2)
            else:
                # Produit connu : Formule hybride complète + Bonus/Malus des avis clients (+10% par Like)
                bonus_feedback = scores_feedback_global.get(idx, 0) * 0.1
                score_final = (score_texte * 0.5) + (note_normalisee * 0.2) + (score_social * 0.3) + bonus_feedback

            # --- APPLICATION DES RÈGLES FINANCIÈRES (Si le mode Business est actif) ---
            if activer_business:
                stock_actuel = df.iloc[idx]['stock']
                marge_actuelle = df.iloc[idx]['marge']

                # Pénalité Fatale : Rupture de stock = Recommandation Annulée
                if stock_actuel == 0:
                    score_final = score_final * 0.0
                # Bonus Rentabilité : Produit très margé = Boost IA (+20%)
                elif marge_actuelle == 'Élevée':
                    score_final = score_final * 1.20
                # Malus Faible Rentabilité = Pénalité légère (-10%)
                elif marge_actuelle == 'Basse':
                    score_final = score_final * 0.90

            # Sauvegarde des données complètes pour générer le Radar plus tard
            recommandations_calculees.append((idx, score_final, score_texte, score_social, note_normalisee))

    # Tri du meilleur score au pire
    recommandations_calculees.sort(key=lambda x: x[1], reverse=True)
    # Conservation du Top 3
    top_3 = recommandations_calculees[:3]

    # ==========================================
    # 7. AFFICHAGE FINAL (TEST A/B & RADAR)
    # ==========================================
    st.divider()

    mode_ab_test = st.checkbox("⚖️ Activer le mode Comparaison (Ancien Modèle vs Nouveau Modèle IA)")

    if mode_ab_test:
        # MODE DUEL : Permet de démontrer l'évolution de l'algorithme
        st.markdown("### 🥊 Le Match des Algorithmes")
        col_ancien, col_nouveau = st.columns(2)

        with col_ancien:
            st.error("🤖 Ancien Modèle (Mots exacts uniquement - TF-IDF)")
            vecteur_panier_ancien = np.asarray(matrice_mots[indices_panier].mean(axis=0)).reshape(1, -1)
            scores_anciens = cosine_similarity(vecteur_panier_ancien, matrice_mots).flatten()
            reco_anciennes = [(idx, scores_anciens[idx]) for idx in range(len(df)) if idx not in indices_panier]
            reco_anciennes.sort(key=lambda x: x[1], reverse=True)
            for data_reco in reco_anciennes[:3]:
                prod = df.iloc[data_reco[0]]
                st.info(f"**{prod['produit']}**")
                st.caption(prod['description'])

        with col_nouveau:
            st.success("🧠 Nouveau Modèle (Sémantique + Historique + Business)")
            for data_reco in top_3:
                prod = df.iloc[data_reco[0]]
                st.success(f"**{prod['produit']}**")
                st.caption(prod['description'])

    else:
        # MODE STANDARD : Affichage des recommandations avec Data Visualisation
        st.subheader("💡 TEKLEVELUP vous suggère :")
        cols = st.columns(3)
        for i, data_reco in enumerate(top_3):
            # Extraction des données calculées pour le produit
            idx_reco = data_reco[0]
            score_final = data_reco[1]
            score_texte = data_reco[2]
            score_social = data_reco[3]
            note_normalisee = data_reco[4]
            produit_reco = df.iloc[idx_reco]

            # EXPLICABILITÉ : L'IA identifie les mots-clés communs entre le panier et la suggestion
            mots_recommandes = set(produit_reco['super_description'].lower().split())
            mots_communs = mots_panier.intersection(mots_recommandes)
            mots_communs_propres = mots_communs - set(mots_inutiles)

            with cols[i]:
                st.success(f"**{produit_reco['produit']}**")
                st.caption(f"Catégorie : {produit_reco['categorie']}")

                # Affichage des métriques Business dynamiques
                couleur_stock = "🔴 RUPTURE" if produit_reco['stock'] == 0 else f"🟢 Stock: {produit_reco['stock']}"
                st.markdown(f"*{couleur_stock} | 💰 Marge : {produit_reco['marge']}*")

                st.write(produit_reco['description'])
                st.caption(f"⭐️ Note : {produit_reco['note_sur_5']}/5")
                st.write(f"🔢 Score mathématique : **{score_final:.3f}**")

                # --- LE RADAR DE COMPATIBILITÉ (Spyder Chart avec Plotly) ---
                # Affiche visuellement la répartition de la puissance du score
                categories = ['Sémantique IA', 'Communauté', 'Avis Clients']
                valeurs = [score_texte * 100, score_social * 100, note_normalisee * 100]
                df_radar = pd.DataFrame(dict(Score=valeurs, Axe=categories))
                fig = px.line_polar(df_radar, r='Score', theta='Axe', line_close=True, template="plotly_dark",
                                    height=220)
                fig.update_traces(fill='toself', line_color='#00ff00', fillcolor='rgba(0, 255, 0, 0.2)')
                fig.update_layout(margin=dict(l=25, r=25, t=20, b=20), paper_bgcolor="rgba(0,0,0,0)")

                st.plotly_chart(fig, use_container_width=True)

                # Affichage des badges de mots-clés
                if mots_communs_propres:
                    mots_badges = " ".join([f"`{mot.capitalize()}`" for mot in mots_communs_propres])
                    st.markdown(f"**Points communs :** {mots_badges}")

                # Zone d'interaction Utilisateur (Feedback)
                c1, c2 = st.columns(2)
                if c1.button("👍", key=f"like_{idx_reco}"):
                    enregistrer_feedback(idx_reco, "like")
                    st.rerun()  # Actualisation en direct
                if c2.button("👎", key=f"dislike_{idx_reco}"):
                    enregistrer_feedback(idx_reco, "dislike")
                    st.rerun()

    # ==========================================
    # 8. DASHBOARD ADMINISTRATEUR (Barre Latérale)
    # ==========================================
    with st.sidebar:
        st.header("⚙️ Espace Admin")
        st.write("Performances de l'algorithme en temps réel :")

        if len(top_3) > 0:
            # Calcul de la précision : Combien de produits ont un score jugé "pertinent" (> 0.5)
            nb_pertinents = sum(1 for data in top_3 if data[1] > 0.5)
            precision_at_3 = (nb_pertinents / len(top_3)) * 100

            # Simulation d'un Taux de Clic basé sur le score de pertinence moyen
            score_moyen = sum(data[1] for data in top_3) / len(top_3)
            ctr_simule = score_moyen * 15.0

            st.metric(label="🎯 Précision du Top 3", value=f"{precision_at_3:.0f} %")
            st.metric(label="🖱️ Taux de Clic (CTR) Estimé", value=f"{ctr_simule:.1f} %")
        else:
            st.info("Ajoutez des produits pour voir les métriques.")

        st.divider()
        st.caption("Pondération de l'IA Hybride :")
        st.caption("🧠 Sémantique (Embeddings) : 50%")
        st.caption("🛒 Historique d'achats : 30%")
        st.caption("⭐️ Notes clients : 20%")

    # ==========================================
    # 9. LE KNOWLEDGE GRAPH (Cartographie globale de l'IA)
    # ==========================================
    st.divider()
    st.subheader("🌌 La Galaxie TEK LEVELUP (Knowledge Graph)")
    st.write("Explorez visuellement comment l'IA regroupe et comprend vos produits.")
    afficher_graphe = st.checkbox("🔭 Activer le télescope IA (Générer la carte)")

    if afficher_graphe:
        with st.spinner("L'IA dessine les constellations..."):
            G = nx.Graph()  # Initialisation du réseau

            # Création des points (Nœuds) de la galaxie
            for index, row in df.iterrows():
                G.add_node(index, label=row['produit'], title=row['description'], group=row['categorie'], size=20)

            # Calcul des ponts (Edges) mathématiques entre TOUS les produits
            matrice_similarite_globale = cosine_similarity(matrice_ia)
            seuil_similarite = 0.30  # Tolérance de connexion de l'IA

            for i in range(len(df)):
                for j in range(i + 1, len(df)):
                    score = matrice_similarite_globale[i][j]
                    if score > seuil_similarite:
                        epaisseur = (float)(score - seuil_similarite) * 10
                        G.add_edge(int(i), int(j), value=epaisseur, title=f"Similarité: {score * 100:.0f}%")

            # Configuration de la physique visuelle
            net = Network(height='600px', width='100%', bgcolor='#0E1117', font_color='white')
            # Les particules s'attirent (si similaires) mais se repoussent physiquement pour être cliquables
            net.repulsion(node_distance=150, spring_length=200)

            # Injection et affichage HTML interactif
            net.from_nx(G)
            net.save_graph('galaxie_teklevelup.html')
            HtmlFile = open('galaxie_teklevelup.html', 'r', encoding='utf-8')
            components.html(HtmlFile.read(), height=620)