import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image, ImageDraw

def model_prediction(test_image):
    model_keras_path = r"C:\essat\projet ia\trained_model.keras"
    model_h5_path = r"C:\essat\projet ia\trained_model.h5"

    if os.path.exists(model_keras_path):
        model_path = model_keras_path
    elif os.path.exists(model_h5_path):
        model_path = model_h5_path
    else:
        raise FileNotFoundError("Modèle non trouvé.")

    model = tf.keras.models.load_model(model_path)

    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0)

    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    confidence = float(prediction[0][result_index])
    return result_index, confidence

def annotate_image(image_file, label):
    image = Image.open(image_file).convert("RGB").resize((256, 256))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 220, 256, 256], fill=(0, 0, 0, 180))
    draw.text((10, 225), label, fill="white")
    return image

class_name = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

def get_solution(disease_name):
    solutions = {
        'Apple___Apple_scab': "Utilisez des fongicides et retirez les feuilles infectées.",
        'Apple___Black_rot': "Taillez les branches infectées et appliquez des fongicides.",
        'Apple___Cedar_apple_rust': "Éliminez les arbres hôtes proches et appliquez des fongicides.",
        'Apple___healthy': "Aucune action nécessaire, la plante est en bonne santé.",
        'Blueberry___healthy': "Aucune action nécessaire, la plante est en bonne santé.",
        'Cherry_(including_sour)___Powdery_mildew': "Appliquez des fongicides spécifiques au mildiou poudreux.",
        'Cherry_(including_sour)___healthy': "Aucune action nécessaire, la plante est en bonne santé.",
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': "Utilisez des fongicides et des variétés résistantes.",
        'Corn_(maize)___Common_rust_': "Appliquez des fongicides et surveillez les conditions humides.",
        'Corn_(maize)___Northern_Leaf_Blight': "Utilisez des variétés résistantes et appliquez des fongicides.",
        'Corn_(maize)___healthy': "Aucune action nécessaire, la plante est en bonne santé.",
        'Grape___Black_rot': "Taillez les parties infectées et appliquez des fongicides.",
        'Grape___Esca_(Black_Measles)': "Éliminez les vignes infectées et appliquez des traitements préventifs.",
        'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': "Appliquez des fongicides et améliorez la circulation de l'air.",
        'Grape___healthy': "Aucune action nécessaire, la plante est en bonne santé.",
        'Orange___Haunglongbing_(Citrus_greening)': "Éliminez les arbres infectés et contrôlez les insectes vecteurs.",
        'Peach___Bacterial_spot': "Appliquez des bactéricides et évitez les éclaboussures d'eau.",
        'Peach___healthy': "Aucune action nécessaire, la plante est en bonne santé.",
        'Pepper,_bell___Bacterial_spot': "Appliquez des bactéricides et évitez les éclaboussures d'eau.",
        'Pepper,_bell___healthy': "Aucune action nécessaire, la plante est en bonne santé.",
        'Potato___Early_blight': "Appliquez des fongicides et retirez les feuilles infectées.",
        'Potato___Late_blight': "Utilisez des fongicides systémiques et surveillez les conditions humides.",
        'Potato___healthy': "Aucune action nécessaire, la plante est en bonne santé.",
        'Raspberry___healthy': "Aucune action nécessaire, la plante est en bonne santé.",
        'Soybean___healthy': "Aucune action nécessaire, la plante est en bonne santé.",
        'Squash___Powdery_mildew': "Appliquez des fongicides spécifiques au mildiou poudreux.",
        'Strawberry___Leaf_scorch': "Taillez les feuilles infectées et appliquez des fongicides.",
        'Strawberry___healthy': "Aucune action nécessaire, la plante est en bonne santé.",
        'Tomato___Bacterial_spot': "Appliquez des bactéricides et évitez les éclaboussures d'eau.",
        'Tomato___Early_blight': "Appliquez des fongicides et retirez les feuilles infectées.",
        'Tomato___Late_blight': "Utilisez des fongicides systémiques et surveillez les conditions humides.",
        'Tomato___Leaf_Mold': "Améliorez la circulation de l'air et appliquez des fongicides.",
        'Tomato___Septoria_leaf_spot': "Appliquez des fongicides et retirez les feuilles infectées.",
        'Tomato___Spider_mites Two-spotted_spider_mite': "Utilisez des acaricides pour contrôler les acariens.",
        'Tomato___Target_Spot': "Appliquez des fongicides et surveillez les conditions humides.",
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus': "Contrôlez les insectes vecteurs et utilisez des variétés résistantes.",
        'Tomato___Tomato_mosaic_virus': "Éliminez les plantes infectées et désinfectez les outils.",
        'Tomato___healthy': "Aucune action nécessaire, la plante est en bonne santé."
    }
    return solutions.get(disease_name, "Solution non disponible.")

st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Choisir une page", ["Accueil", "À propos", "Reconnaissance de maladie"])

if app_mode == "Accueil":
    st.header("MUTUAL HACK BY MAE")
    st.subheader("Détection des maladies des plantes par IA - ESSAT")
    st.markdown("📷 Téléversez une image pour identifier la maladie.")

elif app_mode == "À propos":
    st.header("À propos du Dataset")
    st.markdown("📦 Dataset de 87K images, 38 classes, source : Kaggle")

elif app_mode == "Reconnaissance de maladie":
    st.header("Détection de Maladie")

    # Ajouter des jauges pour les paramètres environnementaux
    st.subheader("🌡️ Paramètres Environnementaux")
    
    # Température
    temperature = 32  # Valeur fictive
    st.metric(label="Température", value=f"{temperature}°C")
    
    # Humidité
    humidity = 40  # Valeur fictive
    st.metric(label="Humidité", value=f"{humidity}%")
    
    # Humidité du sol
    soil_moisture = 30  # Valeur fictive
    st.metric(label="Humidité du sol", value=f"{soil_moisture}%")
    
    # Message pour l'irrigation
    if soil_moisture < 50:  # Exemple de seuil
        st.warning("💧 La zone est irriguée.")
    else:
        st.success("✅ La zone n'a pas besoin d'irrigation.")

    # Téléversement de l'image pour la reconnaissance de maladie
    test_image = st.file_uploader("Choisissez une image :", type=["jpg", "jpeg", "png"])

    if test_image is not None:
        st.image(test_image, caption="Image importée", use_column_width=True)

        if st.button("Prédire"):
            with st.spinner("Chargement..."):
                try:
                    result_index, confidence = model_prediction(test_image)
                    disease_name = class_name[result_index]
                    solution = get_solution(disease_name)

                    st.success(f"✅ Maladie détectée : {disease_name}")
                    st.info(f"🔬 Confiance du modèle : {confidence*100:.2f}%")

                    annotated = annotate_image(test_image, disease_name)
                    st.image(annotated, caption="Annotation", use_column_width=True)

                    with st.expander("ℹ️ En savoir plus"):
                        st.markdown(f"💊 Solution recommandée : {solution}")

                    if 'history' not in st.session_state:
                        st.session_state.history = []
                    st.session_state.history.append({
                        'Maladie': disease_name,
                        'Confiance (%)': round(confidence * 100, 2),
                        'Image': test_image.name
                    })

                except Exception as e:
                    st.error(f"Erreur : {e}")

        if 'history' in st.session_state:
            st.subheader("📊 Historique des prédictions")
            st.dataframe(st.session_state.history)
