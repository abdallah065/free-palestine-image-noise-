import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Define the create_adversarial_pattern function
def create_adversarial_pattern(input_image, input_label):
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = pretrained_model(input_image)
        loss = loss_object(input_label, prediction)

    gradient = tape.gradient(loss, input_image)
    signed_grad = tf.sign(gradient)
    return signed_grad

st.title("Adversarial Image Generation")

# Generate a unique key using the current timestamp
key = int(time.time())

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"], key=key)

if uploaded_image is not None:
    image = Image.open(uploaded_image)

    # Load the MobileNetV2 model
    pretrained_model = tf.keras.applications.MobileNetV2(weights='imagenet')

    # Preprocess the image
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = image[tf.newaxis, ...]

    # Choose a target class index (e.g., 0 for the first class)
    target_class_index = 0
    label = tf.one_hot(target_class_index, image.shape[-1])

    # Define the loss function (Categorical Crossentropy in this case)
    loss_object = tf.keras.losses.CategoricalCrossentropy()

    epsilons = [0.07]

    for eps in epsilons:
        adv_x = image + eps * create_adversarial_pattern(image, label)
        adv_x = tf.clip_by_value(adv_x, -1, 1)

        # Save the proceeded image to a file
        tf.keras.preprocessing.image.save_img('proceedimage.png', adv_x[0] * 0.5 + 0.5)

        # Create a download link for the proceeded image
        st.markdown('[Download Adversarial Image](proceedimage.png)')
