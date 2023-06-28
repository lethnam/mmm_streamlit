'''
Visualize MMM results with streamlit
'''
import os
import streamlit as st
from scripts.mmmstreamlit import MMMStreamlit
from scripts.utils import generate_demo_model, save_model
st.set_page_config(page_title='MMM Streamlit Home Page', layout='wide')


def cache_and_show(obj: MMMStreamlit = None):
    '''
    Cache the object and display a view on the generated data
    '''
    if obj is not None:
        # Save the object to load in sub-pages
        st.session_state['st_obj'] = obj
        # Show a view on the train data
        obj.show_data()


if st.button('Generate a new demo model'):
    # Generate and cache the model
    mmm_model = generate_demo_model()
    st.session_state['mmm_model'] = mmm_model
    st.write('A model has been generated')
    # Create and cache the Streamlit object
    st_obj = MMMStreamlit(mmm_model)
    cache_and_show(st_obj)
    st.session_state['new_model_generated'] = 1

if 'new_model_generated' in st.session_state and st.session_state['new_model_generated'] == 1 and st.button('Save trained demo model'):
    mmm_model = st.session_state['mmm_model']
    st.subheader(type(mmm_model))
    save_model(mmm_model)
    st.write('Model has been saved')

if st.button('Load pre-trained demo model'):
    if os.path.isfile('results/model.pkl'):
        mmm_model = MMMStreamlit.load_model('results/model.pkl')
        st_obj = MMMStreamlit(mmm_model)
        cache_and_show(st_obj)
    else:
        st.write('No model is found. Please generate a demo model')
