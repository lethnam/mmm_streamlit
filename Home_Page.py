'''
Visualize MMM results with streamlit
'''
import os
import streamlit as st
from scripts.mmmstreamlit import MMMStreamlit
from scripts.utils import generate_demo_model, save_model
st.set_page_config(layout='wide')


def save_and_show(obj: MMMStreamlit = None):
    if obj is not None:
        # Save the object to load in sub-pages
        st.session_state['st_obj'] = obj
        # Show a view on the train data
        obj.show_data()


if st.button('Generate a new demo model'):
    mmm_model = generate_demo_model()
    st_obj = MMMStreamlit(mmm_model)
    save_and_show(st_obj)
    if st.button('Save trained demo model'):
        save_model(mmm_model)

if st.button('Load pre-trained demo model'):
    if os.path.isfile('results/model.pkl'):
        mmm_model = MMMStreamlit.load_model('results/model.pkl')
        st_obj = MMMStreamlit(mmm_model)
        save_and_show(st_obj)
    else:
        st.write('No model is found. Please generate a demo model')
