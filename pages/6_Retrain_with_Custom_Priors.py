import streamlit as st
st.set_page_config(page_title='Custom Priors', layout='wide')

if 'st_obj' in st.session_state:
    # Retreive the cached model object
    st_obj = st.session_state['st_obj']

    # Set custom priors
    st.info("LightweightMMM's default priors for Hill function is Gamma(1, 1) (https://lightweight-mmm.readthedocs.io/en/latest/custom_priors.html)")
    st.subheader('Set custom priors for the Hill function')
    st.write('Gamma dist priors for "half_max_effective_concentration"')
    st.number_input('mean', value=1,  key='halfmax_mean')
    st.number_input('rate', value=1, key='halfmax_rate')
    st.write('Gamma dist priors for "slope"')
    st.number_input('mean', value=1, key='slope_mean')
    st.number_input('rate', value=1, key='slope_rate')

    # Retrain the model
    if all(i in st.session_state for i in ['halfmax_mean', 'halfmax_rate', 'slope_mean', 'slope_rate']):
        if st.button('Retrain the model'):
            halfmax_priors = {'half_max_effective_concentration': {'concentration': st.session_state['halfmax_mean'],
                                                                   'rate': st.session_state['halfmax_rate']}}
            slope_priors = {'slope': {'concentration': st.session_state['slope_mean'],
                                      'rate': st.session_state['slope_rate']}}
            mmm_model = st_obj.mmm_model
            mmm_model.set_custom_priors(
                {**halfmax_priors, **slope_priors})
            mmm_model.train(n_warmup=500, n_samples=500, n_chains=1)
            mmm_model.get_diagnostics()
            mmm_model.plot_media_effects()
            # Cache
            st_obj.mmm_model = mmm_model
            st.session_state['st_obj'] = st_obj
            st.info('The model is retrained. All pages are updated.')
else:
    st.info('No pre-trained model found. Go to Home Page to load, or generate a demo')
