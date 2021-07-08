from skimage import io
import onnxruntime as ort
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from skimage import img_as_float
from utils_face import fix_eyes, get_face_aligned
import logging

pet_parser = {
    1: 'cat',
    2: 'dog'
}

@st.cache
def load_models():
    ##  OBJ DETECTION MODEL
    det_model = ort.InferenceSession("00_models/00_pet_detection.onnx")

    ## KPT DETECTION MODEL
    kpt_model = ort.InferenceSession("00_models/01_kpts_detection.onnx")

    ## FACE COMPARISON MODEL
    catface_model = ort.InferenceSession("00_models/02_catfacenet.onnx")
    dogface_model = ort.InferenceSession("00_models/02_dogfacenet.onnx")
    return det_model, kpt_model, catface_model, dogface_model


def parse_labels(labels, parser=pet_parser):
    return [parser[l] for l in labels]

@st.cache
def get_OD_results(img_np, det_model, box_thresh=0.8):
    logger = logging.getLogger(__name__ + ":OD")
    h, w, _ = img_np.shape
    input = img_np[np.newaxis, ...]

    detections = det_model.run(["detection_boxes", "detection_scores", "detection_classes"],
                               {'input_tensor': input.astype(np.uint8)})
    logger.debug("Got pet detections")
    boxes = detections[0][0]
    scores = detections[1][0]
    labels = detections[2][0]

    boxes = boxes[scores >= box_thresh]
    boxes = (boxes * [h, w, h, w]).astype(int)
    labels = labels[scores >= box_thresh]
    logger.debug("returning filtered pet detections")
    return boxes, labels


def preprocess4kpts(img_pil, boxes, size=(224, 224)):
    logger = logging.getLogger(__name__ + ":preprocessingKPTS")
    if len(boxes) > 0:
        logger.debug("found pets!")
        pets = list()
        for box in boxes:
            top, left, bottom, right = box
            pet_i = img_pil.crop((left, top, right, bottom))
            pet_i = pet_i.resize(size, resample=Image.BILINEAR)
            pet_i = np.array(pet_i)
            pet_i = pet_i.transpose(2, 0, 1)
            pet_i = pet_i[np.newaxis, ...]
            pets.append({'pet': pet_i, 'top': top, 'left': left, 'h': bottom-top,'w': right-left})
        logger.debug("processed all pets")
        return pets
    else:
        logger.debug("no pets found")
        return list()


def get_kpts(img_np, kpt_model, shape=(1, 3, 224, 224)):
    logger = logging.getLogger(__name__ + ":getKPTS")
    if img_np.shape == shape:
        logger.debug("correct shape")
        if img_np.dtype == np.uint8:
            model_input = img_as_float(img_np)
        else:
            model_input = img_np
        kpts = kpt_model.run(None, {'input': model_input.astype(np.float32)})[0][0]
        # normalization of kpts
        kpts = (kpts + 1) * (224 / 2)
        kpts = np.array([[kpts[i-1], kpts[i]] for i in range(1, 6, 2)])
        logger.debug("got kpts")
    else:
        logger.debug("incorrect shape, empty response")
        kpts = np.array([[]])
    return kpts

def postprocess4kpts(pets, kpts, size=(224, 224)):
    logger = logging.getLogger(__name__ + ":postprocessingKPTS")
    if len(pets) > 0:
        logger.debug("found pets!")
        new_kpts = list()
        for i, pet in enumerate(pets):
            top, left, h, w = pet['top'], pet['left'], pet['h'], pet['w']
            kpt_i = kpts[i] * [w / size[0], h / size[1]]
            kpt_i = kpt_i + [left, top]
            new_kpts.append(kpt_i)

        logger.debug("processed all pets")
        return new_kpts
    else:
        logger.debug("no pets found")
        return kpts

def preprocess4embedding(img_np, labels, pet_type, kpts, size=(224,224,3)):
    faces, labelss = list(), list()
    for i, kpt in enumerate(kpts):
        leye, reye, nose = fix_eyes(kpt[0], kpt[1], kpt[2], num_eyes=2)
        face_allgnd = get_face_aligned(img_np, leye, reye, nose, labels[i], SIZE=size)
        if labels[i] in pet_type:
            faces.append(face_allgnd)
            labelss.append(labels[i])
    return np.array(faces), labelss


def get_embedding(img_np, emb_model, shape=(1, 224, 224, 3)):
    logger = logging.getLogger(__name__ + ":getEmb")
    if img_np.shape == shape:
        logger.debug("correct shape")
        return emb_model.run(None, {'input_1': img_np.astype(np.float32)})[0]
    else:
        logger.debug("incorrect shape, empty response: " + str(img_np.shape))
        return np.empty((1,32))

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