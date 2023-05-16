import streamlit as st
import tensorflow as tf

def main():
    ConvNet = tf.keras.models.load_model('model/ConvNet')
    
    st.title("Welcome to ConvNet")
    st.write("This is just a simple webpage, where you can test out a model's ability to identify an image.")
    uploaded_image = st.file_uploader("Please upload an image here!")
    if uploaded_image is not None:
        image = 