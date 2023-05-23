import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import urllib.request
import cv2
import tensorflow as tf
from funcs import *
import streamlit.components.v1 as components

def main():

    st.set_page_config(layout="wide", initial_sidebar_state='expanded')

    page_options = ["Find similar items",
                    "Customer Recommendations",
                    "Product Captioning"]
    


    
    page_selection = st.sidebar.radio("Try", page_options)

    articles_df = pd.read_csv('articles.csv')
    
    models = ['Similar items based on image embeddings', 
              'Similar items based on text embeddings', 
              'Similar items based discriptive features', 
              'Similar items based on embeddings from TensorFlow Recommendrs model',
              'Similar items based on a combination of all embeddings']
    
    model_descs = ['Image embeddings are calculated using VGG16 CNN from Keras', 
                  'Text description embeddings are calculated using "universal-sentence-encoder" from TensorFlow Hub',
                  'Features embeddings are calculated by one-hot encoding the descriptive features provided by H&M',
                  'TFRS model performes a collaborative filtering based ranking using a neural network', 
                  'A concatenation of all embeddings above is used to find similar items']
        
    imgURL = st.sidebar.text_input('Image path', '')
    my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if my_upload is not None:
        print(my_upload.name)
        image = Image.open(my_upload)
        st.sidebar.image(image)
        imgURL = ''

    if imgURL is not None:
        path = "input." + imgURL.split('.')[-1]
        try: 
            urllib.request.urlretrieve(imgURL, path)
            st.write('The current image is', path)
            image = Image.open(path)
            st.sidebar.image(image)
        except:
            st.write('Cannot download')