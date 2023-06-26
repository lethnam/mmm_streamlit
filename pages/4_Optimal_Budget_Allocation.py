import streamlit as st
st.set_page_config(layout='wide')

if 'st_obj' in st.session_state:
    st_obj = st.session_state['st_obj']
    st_obj.optimization_plots()
else:
    st.subheader('No data found')
