import streamlit as st
st.set_page_config(page_title='EDA on the train data', layout='wide')

if 'st_obj' in st.session_state:
    # Retreive the cached model object
    st_obj = st.session_state['st_obj']

    # Plots
    st_obj.eda_plots()

else:
    st.info('No data found. Go to Home Page to load, or generate a demo')
