import streamlit as st
st.set_page_config(layout='wide')


if 'st_obj' in st.session_state:
    st_obj = st.session_state['st_obj']
    # Fields to insert budget and unit prices
    budget = st.number_input('Insert a budget')
    unit_prices = {}
    for i in st_obj.mmm_model.media_vars:
        unit_prices[i] = st.number_input(f'Insert unit price of {i}')

    # Run optimization and plots
    if budget != 0 and all(v != 0 for v in unit_prices.values()):
        st.subheader(
            f'Allocating {budget} with unit price {list(unit_prices.values())}')
        st_obj.mmm_model.run_optimization(st_obj,
                                          n_time_period=len(
                                              st_obj.mmm_model.extra_features_test),
                                          budget=budget,
                                          prices=list(unit_prices.values()),
                                          extra_features=st_obj.mmm_model.extra_features_test)
        st_obj.optimization_plots()
else:
    st.subheader('No data found')
