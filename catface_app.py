import streamlit as st
import tensorflow as tf
import boto3
import pandas as pd
import json
import numpy as np

from skimage import io, transform

from loss_utils import triplet, triplet_acc
# from utils import load_images

import pickle


st.set_option('deprecation.showfileUploaderEncoding', False)
st.write("""
# Test CatFac App
This app is a *test*, please select an image from the dropdown menu at the left!
""")

#st.header("Image Uploaded")

st.sidebar.header('User Input')

#st.sidebar.markdown("""
#[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
#""")

# Collects user input features into dataframe
#uploaded_file = st.sidebar.file_uploader("Upload your input image file", type=["png",'jpg','jpeg'])

json_path = 'db_cat_test.json'
default_img_path = 'https://petfacebucket.s3.amazonaws.com'
test_distance = 0.6
folder_path = 'cat_test'

s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID,
         aws_secret_access_key= AWS_SECRET_ACCESS_KEY)

def file_selector(folder_path='cat_test'):
    filenames = s3.list_objects_v2(Bucket='petfacebucket', Prefix='cat_test/')['Contents']
    filenames = [x['Key'] for x in filenames if x['Key']!= 'cat_test/']
    filenames = [x.split("/")[-1] for x in filenames]
    selected_filename = st.sidebar.selectbox('Select a file for test', filenames)
    return selected_filename


filename = file_selector(folder_path)
st.subheader("You selected: {}".format(filename.split(".")[-3]))
#model_path = 'https://petfacebucket.s3.amazonaws.com/catfacenetEffNet.20200829.ckpt-final.h5'
model_path = 'catfacenetEffNet.20200829.ckpt-final.h5'
model = tf.keras.models.load_model(
            model_path,
            custom_objects={'triplet':triplet, 'triplet_acc':triplet_acc})

with open(json_path, "r") as jsonfile:
  data = json.load(jsonfile)


if filename is not None:

    file_path = default_img_path + "/" + folder_path + "/" + filename
    img = io.imread(file_path)

    if img.shape != (224,224,3):
        img = transform.resize(img, output_shape=(224,224,3))
    st.image(img, width=224)
    emb = model(img)

    df_dist = pd.DataFrame(columns=['img_name', 'distance'])
    for key, values in data.items():
        diff = np.square(emb - values['emb'])
        dist = np.sum(diff, 1)
        df_append = pd.DataFrame({'img_name': [values['filepath']], 'distance': dist})
        df_dist = pd.concat([df_dist, df_append], ignore_index=True)

    df_dist = df_dist.sort_values('distance').reset_index().drop(columns=['index'])
    df_dist['pet_number'] = ["/".join(x[-3:-1]) for x in df_dist.img_name.str.split("/")]
    df_mean_dist = df_dist.groupby('pet_number').distance.agg(['mean', 'count']).reset_index()
    df_mean_dist = df_mean_dist[df_mean_dist['mean'] < test_distance].sort_values('mean').reset_index().drop(columns=['index'])

    #df_dist['filepath'] = default_img_path + df_dist.img_name
    #df_dist = df_dist.sort_values('dist').reset_index().drop(columns=['index'])
    max_imgs = min(5, len(df_mean_dist))

    if len(df_mean_dist) > 0:
        my_bar = st.progress(0)
        st.header("Similar Cats:")
        for i in range(max_imgs):
            best_pet = df_mean_dist.pet_number[i]
            st.subheader("Name: {}".format(best_pet.split("/")[-1]))
            best_dist = df_mean_dist['mean'][i]
            best_pet_path = default_img_path + "/" + best_pet
            #list_imgs = [x for x in os.listdir(best_pet_path) if ".DS" not in x]
            #sample_img = np.random.choice(list_imgs)
            sample_img = df_dist[df_dist.pet_number==best_pet].img_name.values[0]

            best_img_path = default_img_path + "/" + sample_img
            best_img = io.imread(best_img_path)

            my_bar.progress(1/max_imgs * (i+1) )

            st.image(best_img, width=224)
            st.write("Similarity: {:2f}%".format((1-(best_dist/(test_distance*4)))*100))
    else:
        st.write("No similar image found")

else:
    st.write("No Image Selected")


# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
#penguins_raw = pd.read_csv('penguins_cleaned.csv')
#penguins = penguins_raw.drop(columns=['species'])
#df = pd.concat([input_df,penguins],axis=0)

# Encoding of ordinal features
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
#encode = ['sex','island']
#for col in encode:
#    dummy = pd.get_dummies(df[col], prefix=col)
#    df = pd.concat([df,dummy], axis=1)
#    del df[col]
#df = df[:1] # Selects only the first row (the user input data)

# Displays the user input features
#st.subheader('User Input features')

#if uploaded_file is not None:
#    st.write(df)
#else:
#    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
#    st.write(df)

# Reads in saved classification model
#load_clf = pickle.load(open('penguins_clf.pkl', 'rb'))

# Apply model to make predictions
#prediction = load_clf.predict(df)
#prediction_proba = load_clf.predict_proba(df)

#st.subheader('Prediction')
#penguins_species = np.array(['Adelie','Chinstrap','Gentoo'])
#st.write(penguins_species[prediction])

#st.subheader('Prediction Probability')
#st.write(prediction_proba)