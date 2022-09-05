import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats
from scipy.stats import norm
import altair as alt

st.set_page_config(
    page_title="Vietnamese Image Captioning", page_icon="📊", initial_sidebar_state="expanded"
)

st.write(
    """
# 📊 A/B Testing App
Upload your experiment results to see the significance of your A/B test.
"""
)

# uploaded_file = st.file_uploader("Upload CSV", type=".csv")

# use_example_file = st.checkbox(
#     "Use example file", False, help="Use in-built example file to demo the app"
# )


ab_default = None
result_default = None

# If CSV is not uploaded and checkbox is filled, use values from the example file
# and pass them down to the next if block
# if use_example_file:
#     uploaded_file = "model_KT/test.jpg"
#     ab_default = ["variant"]
#     result_default = ["converted"]

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
        self.model = {
            'model_KT': {},
            'model_Phu': {}
        }


history = History()


def changeCheckBox():
    print("changeCheckBox")
    # st.experimental_rerun()


print("head")

with st.form("my-form", clear_on_submit=True):
    uploaded_file = st.file_uploader("Upload image", type=['png', 'jpg'])

    ab = st.radio(
        "model type",
        options=["model_KT", "model_Phu"],
        index=0,
        key="model",
        help="TBD",
    )

    #False is checked
    use_all = st.checkbox('use all', value=True, on_change=changeCheckBox())
    not_use_all = st.checkbox('not use all', value=True, on_change=changeCheckBox())

    if use_all is not False and not_use_all is not False:
        st.experimental_rerun()

    # if agree is not False:
    #     options = st.multiselect(
    #         'What are your favorite colors',
    #         ['Green', 'Yellow', 'Red', 'Blue'])

    # if not name:
    #     st.warning('Please input a name.')
    #     st.stop()
    #
    #     st.experimental_rerun()

    # if agree:
    #     print("use all")
    # else:
    #     print("dont use all")

    #
    #     st.write('You selected:', options)

    submitted = st.form_submit_button("UPLOAD!")

if submitted and uploaded_file is not None:
    # do stuff with your uploaded file

    print("uploaded_file")
    print(type(uploaded_file))

    file_location = f"streamlit/tmp/{uploaded_file.name}"
    with open(file_location, "wb+") as file_object:
        file_object.write(uploaded_file.read())
    print(f"info: file {uploaded_file.name} saved at {file_location}")

    # st.markdown("### Data preview")
    # st.image(file_location)

    # type(uploaded_file) == str, means the example file was used
    name = (
        "test.jpg" if isinstance(uploaded_file, str) else uploaded_file.name
    )
    st.write("")
    st.write("## Results for A/B test from ", name)
    st.write("")

    # Obtain the metrics to display
    if ab == 'model_KT':
        print("model type model_KT")
        res, attention_plot = utilsKT.evaluate_load(file_location)
    elif ab == 'model_Phu':
        print("model type model_Phu")
        res = utilsPhu.predict(file_location)

    st.image(file_location, caption=res)

    history.model[ab].update({file_location: res})

else:
    st.warning("Please upload image!")

print("history.model")
print(history.model)

st.write("History")

# with st.expander("History"):
#
#      for

col1, col2 = st.columns(2)

with col1:
    st.header("model_KT")
    for key, value in history.model['model_KT'].items():
        st.image(key, caption=value)

with col1:
    st.header("model_Phu")
    for key, value in history.model['model_Phu'].items():
        st.image(key, caption=value)

# if MyVar == "Whatever value you want to monitor":
#     st.experimental_rerun()
