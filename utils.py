from skimage import io
import streamlit as st
import numpy as np
import pandas as pd

def get_similar_pictures(data, emb, default_img_path, pet_type, test_distance = 0.6):

    df_dist = pd.DataFrame(columns=['img_name', 'distance'])
    for key, values in data.items():
        diff = np.square(emb - values['emb'])
        dist = np.sum(diff, 1)
        df_append = pd.DataFrame({'img_name': [values['filepath']], 'distance': dist})
        df_dist = pd.concat([df_dist, df_append], ignore_index=True)

    df_dist = df_dist.sort_values('distance').reset_index().drop(columns=['index'])
    df_dist['pet_number'] = ["/".join(x[-3:-1]) for x in df_dist.img_name.str.split("/")]
    df_mean_dist = df_dist.groupby('pet_number').distance.agg(['mean', 'count']).reset_index()
    df_mean_dist = df_mean_dist[df_mean_dist['mean'] < test_distance].sort_values('mean').reset_index().drop(
        columns=['index'])

    max_imgs = min(6, len(df_mean_dist))

    if len(df_mean_dist) > 0:
        st.header("Similar {}s:".format(pet_type))
        col1, col2, col3 = st.beta_columns(3)
        for i in range(max_imgs):
            best_pet = df_mean_dist.pet_number[i]
            if i in [0,3]:
                col1.subheader("Name: {}".format(best_pet.split("/")[-1]))
            elif i in [1,4]:
                col2.subheader("Name: {}".format(best_pet.split("/")[-1]))
            else:
                col3.subheader("Name: {}".format(best_pet.split("/")[-1]))
            best_dist = df_mean_dist['mean'][i]
            #list_imgs = [x for x in os.listdir(best_pet_path) if ".DS" not in x]
            #sample_img = np.random.choice(list_imgs)

            list_imgs = df_dist[df_dist.pet_number == best_pet].img_name.values
            sample_img = np.random.choice(list_imgs)
            best_img_path = default_img_path + "/" + sample_img
            best_img = io.imread(best_img_path)

            if i in[0,3]:
                col1.image(best_img, width=224)
                col1.write("Similarity: {:2f}%".format((1 - (best_dist / (test_distance * 4))) * 100))
            elif i in [1,4]:
                col2.image(best_img, width=224)
                col2.write("Similarity: {:2f}%".format((1 - (best_dist / (test_distance * 4))) * 100))
            else:
                col3.image(best_img, width=224)
                col3.write("Similarity: {:2f}%".format((1 - (best_dist / (test_distance * 4))) * 100))
    else:
        st.write("No similar image found, try selecting 'Wrong picture processed' ")