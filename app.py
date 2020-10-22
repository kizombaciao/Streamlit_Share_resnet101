'''
https://github.com/terryz1/Image_Classification_App
https://towardsdatascience.com/create-an-image-classification-web-app-using-pytorch-and-streamlit-f043ddf00c24

'''

import streamlit as st
from PIL import Image
from clf import predict

st.set_option('deprecation.showfileUploaderEncoding', False)

st.title('Resnet 101 ImageNet Classfication App')
st.write('')

file_up = st.file_uploader('Upload an Image', type='jpg')

if file_up is not None:
    image = Image.open(file_up)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write('')
    st.write('Just a second ... ')
    labels = predict(file_up)
    
    for i in labels:
        st.write('Prediction (index, name)', i[0], ', Score: ', i[1])