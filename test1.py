import streamlit as st
import tensorflow as tf
import numpy as np
import os

# Fonction pour charger et prédire l'image avec le modèle
def model_prediction(test_image):
    # Définir les chemins possibles du modèle
    model_keras_path = r"C:\essat\projet ia\trained_model.keras"
    model_h5_path = r"C:\essat\projet ia\trained_model.h5"

    # Vérification de l'existence du modèle
    if os.path.exists(model_keras_path):
        model_path = model_keras_path
    elif os.path.exists(model_h5_path):
        model_path = model_h5_path
    else:
        raise FileNotFoundError("Aucun fichier modèle trouvé (.keras ou .h5). Vérifiez le chemin.")

    # Chargement du modèle
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"✅ Modèle chargé : {model_path}")
    except Exception as e:
        raise RuntimeError(f"Erreur de chargement du modèle : {e}")

    # Prétraitement de l'image
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0)

    # Prédiction
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    confidence = np.max(prediction)
    return result_index, confidence

# Fonction pour obtenir la solution en fonction de la maladie détectée
def get_solution(disease_name):
    disease_data = {
        'Apple___Apple_scab': {
            "nom": "Tavelure du pommier",
            "culture": "Pommier",
            "gravite": "Modérée à élevée",
            "traitement": "Appliquer du captane à 2 g/L toutes les 7 à 14 jours pendant la période de croissance active. Éviter les applications de lime-soufre immédiatement après le captane pour prévenir les dommages foliaires."
        },
        'Apple___Black_rot': {
            "nom": "Pourriture noire du pommier",
            "culture": "Pommier",
            "gravite": "Élevée",
            "traitement": "Utiliser du thiophanate-méthyl à 1,25 kg/ha en pulvérisation foliaire, répéter 3 à 5 fois à intervalles de 7 à 10 jours."
        },
        'Apple___Cedar_apple_rust': {
            "nom": "Rouille cèdre-pommier",
            "culture": "Pommier",
            "gravite": "Modérée",
            "traitement": "Appliquer du myclobutanil ou du propiconazole toutes les 7 à 14 jours, en commençant au stade du bouton rose jusqu'à 10 à 14 jours après la chute des pétales."
        },
        'Apple___healthy': {
            "nom": "Aucune maladie détectée",
            "culture": "Pommier",
            "gravite": "Aucune",
            "traitement": "Aucun traitement nécessaire."
        },
        'Blueberry___healthy': {
            "nom": "Aucune maladie détectée",
            "culture": "Myrtille",
            "gravite": "Aucune",
            "traitement": "Aucun traitement nécessaire."
        },
        'Tomato___Early_blight': {
            "nom": "Brûlure précoce de la tomate",
            "culture": "Tomate",
            "gravite": "Modérée",
            "traitement": "Appliquez des fongicides et retirez les feuilles infectées."
        },
        'Tomato___Late_blight': {
            "nom": "Brûlure tardive de la tomate",
            "culture": "Tomate",
            "gravite": "Élevée",
            "traitement": "Utilisez des fongicides systémiques et surveillez les conditions humides."
        },
        'Tomato___healthy': {
            "nom": "Aucune maladie détectée",
            "culture": "Tomate",
            "gravite": "Aucune",
            "traitement": "Aucun traitement nécessaire."
        },
    }
    return disease_data.get(disease_name, {
        "nom": "Maladie inconnue",
        "culture": "Inconnue",
        "gravite": "Inconnue",
        "traitement": "Aucune solution disponible."
    })

# Liste des classes
class_name = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___healthy'
]

# Interface utilisateur Streamlit
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Choisir une page", ["Accueil", "À propos", "Reconnaissance de maladie"])

# Page d'accueil
if app_mode == "Accueil":
    st.header("MUTUAL HACK BY MAE")
    st.subheader("Système de Reconnaissance de Maladies des Plantes avec CNN - Groupe ESSAT")
    st.markdown("""
    ### Introduction
    Ce système est basé sur un réseau de neurones convolutif entraîné sur un dataset de 87K images de feuilles saines et malades.

    ### Comment utiliser l'application ?
    1. Téléversez une image de feuille.
    2. Cliquez sur "Predict".
    3. Le modèle affiche la maladie détectée.

    ### Modèle
    Entraîné avec Tensorflow/Keras. Précision : 99.5%.

    ### Travaux futurs
    - Apprentissage par transfert
    - Application mobile/web
    """)

# Page À propos
elif app_mode == "À propos":
    st.header("À propos du Dataset")
    st.markdown("""
    Le dataset provient de Kaggle (87K images, 38 classes).

    **Structure :**
    - Train : 70,295 images
    - Valid : 17,572 images
    - Test  : 33 images
    """)

# Page de prédiction
elif app_mode == "Reconnaissance de maladie":
    st.header("Reconnaissance de Maladie")

    # Ajouter des jauges pour les paramètres environnementaux
    st.subheader("Paramètres Environnementaux")
    temperature = 32  # Valeur fictive
    st.metric(label="Température", value=f"{temperature}°C")
    humidity = 40  # Valeur fictive
    st.metric(label="Humidité", value=f"{humidity}%")
    soil_moisture = 30  # Valeur fictive
    st.metric(label="Humidité du sol", value=f"{soil_moisture}%")

    if soil_moisture < 50:
        st.warning("La zone est irriguée.")
    else:
        st.success("La zone n'a pas besoin d'irrigation.")

    # Téléversement de l'image pour la reconnaissance de maladie
    test_image = st.file_uploader("Choisissez une image :", type=["jpg", "jpeg", "png"])

    if test_image is not None:
        if st.button("Afficher l'image"):
            st.image(test_image, use_column_width=True)

        if st.button("Predict"):
            with st.spinner("Chargement et prédiction en cours..."):
                try:
                    result_index, confidence = model_prediction(test_image)
                    if 0 <= result_index < len(class_name):
                        disease_name = class_name[result_index]
                        solution = get_solution(disease_name)
                        st.success(f"Prédiction du modèle : **{disease_name}**")
                        st.info(f"Confiance : {confidence*100:.2f}%")
                        st.info(f"Solution recommandée : {solution}")
                    else:
                        st.error("Erreur : L'index prédit est hors des limites de la liste des classes.")
                except Exception as e:
                    st.error(f"Une erreur est survenue : {e}")
