import streamlit as st
st.set_page_config(page_title='Run budget optimization', layout='wide')


if 'st_obj' in st.session_state:
    # Retreive the cached model object
    st_obj = st.session_state['st_obj']

    # Fields to insert budget and unit prices
    budget = st.number_input('Insert a budget', key='opt_budget')
    unit_prices = {}
    for i in st_obj.mmm_model.media_vars:
        unit_prices[i] = st.number_input(f'Insert unit price of {i}')
    st.session_state['unit_prices'] = unit_prices

    # Run optimization and plots
    if st.session_state['opt_budget'] != 0 and all(v != 0 for v in st.session_state['unit_prices'].values()):
        if st.button('Run optimization'):
            budget = st.session_state['opt_budget']
            unit_prices = st.session_state['unit_prices']
            st.info(
                f'Allocating {budget} with unit prices {list(unit_prices.values())}')
            st_obj.mmm_model.run_optimization(n_time_periods=st_obj.mmm_model.extra_features_test.shape[0],
                                              budget=budget,
                                              prices=list(
                                                  unit_prices.values()),
                                              extra_features_opt=st_obj.mmm_model.extra_features_test)
            st_obj.optimization_plots()
else:
    st.info('No data found. Go to Home Page to load, or generate a demo')
