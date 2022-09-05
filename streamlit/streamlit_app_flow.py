import streamlit as st

st.set_page_config(
    page_title="Vietnamese Image Captioning", page_icon="📊", initial_sidebar_state="expanded"
)

st.write(
    """
# 📊 A/B Testing App
Upload your experiment results to see the significance of your A/B test.
"""
)

import os
import shutil

# for model_KT
from model_KT import utilsHandle as utilsKT

from model_Phu import utils as utilsPhu


@st.cache
def reset_folder():
    print("reset_folder")
    dir_path = os.path.dirname(os.path.realpath(__file__))
    tmpPath = os.path.join(dir_path, 'tmp')
    if os.path.exists(tmpPath):
        shutil.rmtree(tmpPath)
    if not os.path.exists(tmpPath):
        os.mkdir(tmpPath)


reset_folder()


@st.cache(allow_output_mutation=True)
class History:
    def __init__(self):
        self.image = {}


history = History()


def showHistory():
    if len(history.image.keys()) > 0:
        st.header("History")

        for img_path in history.image.keys():
            st.image(img_path)
            for key, value in history.image[img_path].items():
                st.write(key + ": ", value)


print("head")

uploaded_file = st.file_uploader("Upload image", type=['png', 'jpg'])

if uploaded_file is None:
    showHistory()
    st.stop()

# False is checked
use_all = st.checkbox('use all', value=False)

options = []
all_model = {"model_KT": utilsKT.predict, "model_Phu": utilsPhu.predict}

if use_all == False:
    options = st.multiselect(
        'Choose model',
        all_model.keys())
else:
    options = all_model.keys()

if len(options) == 0:
    showHistory()
    st.stop()

if st.button('Predict'):

    # save img
    file_location = f"streamlit/tmp/{uploaded_file.name}"
    with open(file_location, "wb+") as file_object:
        file_object.write(uploaded_file.read())
    print(f"info: file {uploaded_file.name} saved at {file_location}")
    history.image[file_location] = {}
    for model in options:
        # result = {model: all_model[model](file_location)}
        history.image[file_location].update({model: all_model[model](file_location)})

print("history.image")
print(history.image)

showHistory()
