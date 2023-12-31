import streamlit as st
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import tempfile



pretrained_model = tf.keras.applications.MobileNetV2(include_top=True,
                                                     weights='imagenet')
pretrained_model.trainable = False

# ImageNet labels
decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions

# Helper function to preprocess the image so that it can be inputted in MobileNetV2
def preprocess(image):
  image = tf.cast(image, tf.float32)
  image = tf.image.resize(image, (224, 224))
  image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
  image = image[None, ...]
  return image

def preprocess_org(image):
  image = tf.cast(image, tf.float32)
  image = image[None, ...]
  return image

# Helper function to extract labels from probability vector
def get_imagenet_label(probs):
  return decode_predictions(probs, top=1)[0][0]

def create_adversarial_pattern(input_image, input_label):
  with tf.GradientTape() as tape:
    tape.watch(input_image)
    prediction = pretrained_model(input_image)
    loss = loss_object(input_label, prediction)

  # Get the gradients of the loss w.r.t to the input image.
  gradient = tape.gradient(loss, input_image)
  # Get the sign of the gradients to create the perturbation
  signed_grad = tf.sign(gradient)
  return signed_grad
    
st.title("Adversarial Image Generation")
temp_dir = tempfile.TemporaryDirectory()

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if uploaded_image is not None:
    with open(temp_dir.name + '/uploaded_file.jpg', 'wb') as f:
        f.write(uploaded_image.read())

    st.write("Uploaded file saved to:", temp_dir.name + '/uploaded_file.jpg')
    # Convert the image to a NumPy array
    image_raw = tf.io.read_file(temp_dir.name + '/uploaded_file.jpg') 
    image = tf.image.decode_image(image_raw)

    image_original = preprocess_org(image)
    image = preprocess(image)
    image_probs = pretrained_model.predict(image)

    loss_object = tf.keras.losses.CategoricalCrossentropy()

    # Get the input label of the image.
    labrador_retriever_index = 208
    label = tf.one_hot(labrador_retriever_index, image_probs.shape[-1])
    label = tf.reshape(label, (1, image_probs.shape[-1])) 

    perturbations = create_adversarial_pattern(image, label)
    plt.imshow(perturbations[0] * 0.5 + 0.5)  # To change [-1, 1] to [0, 1]

    epsilons = [0.07]
    descriptions = [('Epsilon = {:0.3f}'.format(eps) if eps else 'Input') for eps in epsilons]

    for i, eps in enumerate(epsilons):
        adv_x = image + eps * perturbations
        adv_x = tf.clip_by_value(adv_x, -1, 1)
        im = adv_x
        im =(im[0] * 0.5 + 0.5).numpy() 
        im = Image.fromarray((im * 255).astype('uint8'))
        import io

        # Assuming 'im' contains your adversarial image
        # Convert it to bytes
        image_bytes = io.BytesIO()
        im.save(image_bytes, format='PNG')

        # Create a download button
        st.download_button(
           label="Download Adversarial Image",
           data=image_bytes.getvalue(),
           key="adversarial-image",
           file_name="adversarial_image.png",
           mime="image/png"
         )

     
