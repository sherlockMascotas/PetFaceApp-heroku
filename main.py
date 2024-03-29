import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import pandas as pd
import json
import numpy as np
import os
import io
import boto3
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import requests
import base64
from utils_face import fix_eyes, get_face_aligned

from utils import get_similar_pictures

# For local testing
#from env_variables import set_env_variables

def main():
    # INITIALIZATION
    # For local testing
    #set_env_variables()
    url_base = os.environ['URL_BASE']
    secret_key = os.environ['SECRET_KEY']
    headers = {'Authorization': 'Token '+secret_key}
    proxies = {
        "http": os.environ['QUOTAGUARDSHIELD_URL'],
        "https": os.environ['QUOTAGUARDSHIELD_URL']
    }
    ## SIDEBAR
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.sidebar.header('User Input')

    # Collects user input features into dataframe
    uploaded_file = st.sidebar.file_uploader("Upload your input image file", type=["png", 'jpg', 'jpeg'])
    pet_type = st.sidebar.selectbox(
        "Pet Type:", ("cat", "dog")
    )

    ## S3 Connection
    aws_key = os.environ['AWS_ACCESS_KEY_ID']
    aws_secret = os.environ['AWS_SECRET_ACCESS_KEY']
    s3 = boto3.client('s3', aws_access_key_id=aws_key,
                      aws_secret_access_key=aws_secret)
    default_img_path = 'https://petfacebucket.s3.amazonaws.com'
    folder_path = 'cat_test'

    test_distance = 0.6

    ## LOAD DATABASE
    json_path = '00_db/db_' + pet_type + '.json'
    #    json_path = 'db_cat.json'
    with open(json_path, "r") as jsonfile:
        db = json.load(jsonfile)

    if uploaded_file is not None:
        st.write("""
                # Test PetFace App
                """)

        img_pil = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(img_pil)
        img_np = img_np[:, :, :3]

        url = url_base + 'pet_face'
        buffer = io.BytesIO()
        img_pil.save(buffer, format="JPEG")
        img_b64 = base64.b64encode(buffer.getvalue())
        data = {'pet_type': [pet_type], 'image': img_b64.decode(), 'is_base64': True}
        response = requests.post(url, data=json.dumps(data), headers=headers, proxies=proxies)
        if response.ok:
            boxes = response.json()['boxes']
            kpts = response.json()['kpts']
            labels = response.json()['pet_types']
            embs = response.json()['embs']


            if not st.sidebar.checkbox("Wrong picture processed"):

                if len(boxes) > 0:

                    col1, col2 = st.beta_columns(2)

                    # processing only first pet
                    kpt = np.array(kpts[0])
                    y, x, y2, x2 = boxes[0]

                    fig, ax = plt.subplots(1)
                    ax.imshow(img_np)
                    rect = patches.Rectangle((x, y), x2 - x, y2 - y, linewidth=1, edgecolor='b', facecolor='none')
                    ax.add_patch(rect)
                    ax.plot(kpt[:, 0::2], kpt[:, 1::2], 'o')
                    plt.axis('off')

                    col1.pyplot(fig)
                    col2.subheader(
                        "This is your original picture with eyes and nose detected. Below you'll find your pet face aligned")

                    leye, reye, nose = fix_eyes(kpt[0], kpt[1], kpt[2], num_eyes=2)
                    face_aligned = get_face_aligned(img_np, leye, reye, nose, pet_type)
                    col2.image(face_aligned, use_column_width='auto')
                    col2.subheader(
                        "Does the picture is correctly detected?. Please answer in the sidebar")

                    if pet_type in labels:
                        emb = np.array(embs[0])
                        t = st.empty()
                        t.info("getting similar {}s in our db".format(pet_type))
                        get_similar_pictures(db, emb, default_img_path, pet_type, test_distance=test_distance)
                        t.success("Success!")
                    else:
                        st.write("No {}s found, please check the pet type".format(pet_type))
                else:
                    st.write("No {}s found, please check 'Wrong picture processed to manually annotate'".format(pet_type))

            else:
                st.write("Please help us identify the eyes and the nose of your pet.")
                st.write(
                    "Using the drawing tool at the sidebar select circle and annotate the eyes and nose of your pet (no particular order)")
                st.write("You can also select rectangle from the drawing tool and create a rectangle around your pet head")
                st.write("If necessary use the transform selection at the sidebar to modify your annotations; \n"
                         "aditionaly use the undo, redo and trash buttons near the picture")

                canvas_h = 300
                canvas_w = 600

                drawing_mode = st.sidebar.selectbox(
                    "Drawing tool:", ("circle", "rect", "transform")
                )

                canvas_result = st_canvas(
                    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
                    stroke_width=2,
                    stroke_color='blue',
                    background_color="",
                    background_image=img_pil,
                    update_streamlit=True,
                    drawing_mode=drawing_mode,
                    key="canvas",
                )

                if canvas_result.json_data is not None:
                    df = pd.json_normalize(canvas_result.json_data["objects"])
                    if len(df) > 1:
                        leye, reye, nose = [], [], []
                        kpts = df.loc[df.type == 'circle', ['left', 'top']].values

                        h, w, _ = img_np.shape
                        ratio_factor = (w / h) * (canvas_h / canvas_w)
                        kpts = kpts * [(h / canvas_h) * ratio_factor,
                                       (w / canvas_w) * (1 / ratio_factor)]
                        if len(kpts) <= 2:
                            leye, reye, nose = fix_eyes(kpts[0], kpts[0], kpts[1], num_eyes=1)
                        if len(kpts) > 2:
                            leye, reye, nose = fix_eyes(kpts[0], kpts[1], kpts[2], num_eyes=2)

                        face_aligned = get_face_aligned(img_np, leye, reye, nose, pet_type)

                        col1, col2 = st.beta_columns(2)
                        col2.subheader("This is the picture of your {} aligned!".format(pet_type))
                        col1.image(face_aligned)

                        url_emb = url_base + 'emb'
                        data = {'image':face_aligned.tolist(), 'pet_type':pet_type}

                        response_emb = requests.post(url_emb, data=json.dumps(data), headers=headers, proxies=proxies)
                        if response_emb.ok:
                            emb = np.array(response_emb.json()['emb'])
                            t = st.empty()
                            t.info("getting similar {}s in our db".format(pet_type))
                            get_similar_pictures(db, emb, default_img_path, pet_type, test_distance=test_distance)
                            t.success("Success!")
        else:
            st.write("There was a problem connecting to the server, please try again later")
    else:
        st.write("""
            # Test PetFace App
            This app is a *test*, please upload an image and specify the pet type 
            at the menu on the left
            """)


if __name__ == '__main__':
    main()