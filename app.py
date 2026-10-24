import streamlit as st
from model import load_custom_cnn, load_resnet18
from utils import image_transform, predict

st.set_page_config(page_title="Natural Landscape Classifier", page_icon="🏔️")

st.title("🏔️ Natural Landscape Classifier")
st.write("Upload a natural or urban landscape photo to classify it across 6 categories.")


@st.cache_resource
def get_models():
    return load_custom_cnn(), load_resnet18()


custom_cnn_model, resnet18_model = get_models()

model_choice = st.selectbox(
    label="Select Model Architecture",
    options=["Resnet18", "Custom_CNN"],
)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    transformed_img = image_transform(uploaded_file, model_choice)

    active_model = resnet18_model if model_choice == "Resnet18" else custom_cnn_model
    prediction, confidence = predict(transformed_img, active_model)

    st.success(f"**Prediction:** {prediction} ({confidence:.2f}% confidence)")