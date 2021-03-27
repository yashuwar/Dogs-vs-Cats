import streamlit as st

from PIL import Image, ImageOps

import numpy as np
import tensorflow as tf

st.set_option('deprecation.showfileUploaderEncoding',False)

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model("model.h5")
    return model

model = load_model()

st.markdown("# Dogs vs. Cats prediction web-app", unsafe_allow_html = True)
st.markdown("---", unsafe_allow_html = True)
st.markdown("* This is a simple web-app which predicts whether the image that the user uploads contains a cat or a dog.", unsafe_allow_html = True)
st.markdown("* Connect with me at - https://www.linkedin.com/in/yashwardhan-banta-6566461ab/.", unsafe_allow_html = True)
st.markdown("* GitHub repo (I would appreciate it if you could star it) - https://github.com/yashuwar/Dogs-vs-Cats.",  unsafe_allow_html = True)
st.markdown("---", unsafe_allow_html = True)

st.markdown("# Predictor", unsafe_allow_html = True)
st.markdown("---", unsafe_allow_html = True)
st.markdown("* Upload an image file below and click on the 'Predict' button that appears below the uploaded image to make the predictions.", unsafe_allow_html = True)
uploaded_file = st.file_uploader("Image of cat/dog to be uploaded.", type=['png','jpeg','jpg'])

if uploaded_file is not None:

    st.write("File uploaded! File type: "+uploaded_file.type+".")
    
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
            st.markdown("* ### The image is of a Dog.", unsafe_allow_html = True)
        else:
            st.markdown("* ### The image is of a Cat.", unsafe_allow_html = True)
        
        st.markdown("---", unsafe_allow_html = True)
        st.markdown("### Thank you for using this web-app.", unsafe_allow_html = True)
        st.markdown("---", unsafe_allow_html = True)
    else:
        print("Please click the button to make predictions.")
