import streamlit as st
import tensorflow as tf
import pandas as pd
from PIL import Image, ImageOps
import numpy as np


model = tf.keras.models.load_model("bird_mobile_finetune.h5")
BirdClasses = np.loadtxt("Brirdclass.txt", dtype="<U29", delimiter = '\n')


st.title('Bird 450 species Classification')

st.set_option('deprecation.showfileUploaderEncoding', False)
file = st.file_uploader("Upload the image to be classified", type=["png", "jpg", "jpeg"])

if file is not None:
  u_img = Image.open(file)
  show_image = u_img.resize((600, 400))
  show  = st.image(show_image, 'Uploaded Image', use_column_width=True)
  # We preprocess the image to fit in algorithm.
  #image = np.asarray(u_img) / 255
  #my_image = tf.image.resize(u_img, size = [224, 224])
  #my_image = my_image / 255.
  my_image = u_img.resize((224, 224))
  my_image = np.asarray(my_image) / 255.
  pred = model.predict(tf.expand_dims(my_image, axis=0))
  pred_class = BirdClasses[pred.argmax()]
  top5_pred_percent = np.sort(pred, axis=1)[:, -5:]
  top5_pred_percent = np.flip(top5_pred_percent) * 100
  top5_pred_percent = np.around(top5_pred_percent, decimals=3)
  top5_pred_index = np.argsort(pred, axis=1)[:, -5:]
  top5_pred_index = np.flip(top5_pred_index)
  top5_cate = [BirdClasses[i] for i in top5_pred_index[0]]
  data_res = pd.DataFrame({'Species': top5_cate, 'Confidence Level': top5_pred_percent[0].tolist()})
  st.header("The five most likely bird species")
  st.dataframe(data_res, width = 400, height = 500)

  st.caption("Dataset : https://www.kaggle.com/datasets/gpiosenka/100-bird-species", unsafe_allow_html=True)
  st.caption("Create by SiraH")

