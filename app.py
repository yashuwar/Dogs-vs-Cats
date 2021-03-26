import streamlit as st
import PIL

from PIL import Image, ImageOps


import numpy as np
import tensorflow as tf

st.set_option('deprecation.showfileUploaderEncoding',False)

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model("model.h5")
    return model

model = load_model()

st.title("First app")

st.write("Upload an image file below and click the 'Predict' button to make predictions.")

uploaded_file = st.file_uploader("Image of cat/dog to be uploaded.", type=['png','jpeg','jpg'])

if uploaded_file is not None:

    st.write("File uploaded! File type: ",uploaded_file.type)
    image = Image.open(uploaded_file)
    st.image(image, caption = 'Uploaded file.', use_column_width = True)
    
    bl = st.button("Predict")
    
    if bl:
        
        size = (150, 150)
        
        image = np.asarray(image)
        image = tf.image.resize(image, [150, 150])
        image = np.asarray(image)
        image = np.reshape(image, (1, 150, 150, 3))
        image = image.copy()
        
        image /= 255
        
        label = model.predict_classes(image)
        
        label = label[0][0]
                          
        if label==1:
            st.subheader("Image is of a Dog.")
        else:
            st.subheader("Image is of a Cat.")
        
    else:
        print("Please click the button to make predictions.")
