'''
Visualize MMM results with streamlit
'''
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import streamlit as st
from lightweight_mmm import utils
from scripts.mmm import run as mmm_run
st.set_page_config(layout='wide')


class MMMStreamlit:
    '''
    Streamlit class to display MMM results
    '''

    def __init__(self, mmm_model):
        placeholder = st.empty()
        with placeholder.container():
            st.subheader('Loading MMM results ...')
        self.mmm_model = mmm_model

        self.df_target = self.mmm_model.target
        self.df_media = self.mmm_model.media
        self.df_extra_features = self.mmm_model.extra_features
        self.df_data = pd.concat(
            [self.df_target, self.df_media, self.df_extra_features], axis=1)
        self.media_vars = self.df_media.columns

        placeholder.empty()
        st.subheader('Loaded MMM data and results')

    def show_data(self):
        '''
        Display a view of the train data
        '''
        st.subheader('Train data set')
        st.dataframe(self.df_data.head(100))

    def eda_plots(self):
        '''
        Plots EDA on train data
        '''
        st.header('EDA on train data')

        st.subheader('Correlation heatmap')
        fig_heatmap, ax_heatmap = plt.subplots()
        sns.heatmap(ax=ax_heatmap, data=self.df_data.corr(),
                    annot=True, fmt='.2f')
        st.pyplot(fig=fig_heatmap)

        st.subheader('Target vs. Media')
        st.line_chart(data=self.df_data['target'])
        st.line_chart(data=self.df_data[list(self.media_vars)])

    def posterior_plots(self):
        '''
        Display the posterior plots
        '''
        col1, col2 = st.columns(2)
        with col1:
            st.subheader('Media Posteriors')
            st.pyplot(self.mmm_model.fig_media_posteriors)
        with col2:
            st.subheader('Response Curves')
            st.pyplot(self.mmm_model.fig_response_curves)

    def media_effect_plots(self):
        '''
        Display media effects, ROIs, and media contributions
        '''
        col1, col2 = st.columns(2)
        with col1:
            st.subheader('Media contribution')
            st.pyplot(self.mmm_model.fig_media_effects)
        with col2:
            st.subheader('ROI')
            st.pyplot(self.mmm_model.fig_roi)

        st.subheader('Baseline media contribution')
        st.pyplot(self.mmm_model.fig_media_baseline_contribution)

    def optimization_plots(self):
        '''
        Display pre-/post-optimization budget allocation
        '''
        st.header(
            f'Optimal allocation on {self.mmm_model.opt_budget} extra budget')
        st.pyplot(self.mmm_model.fig_pre_post_optim)


@st.cache_resource
def load_mmm_model():
    # Load MMM results. Run the model if there is no result file
    filepath = 'results/model.pkl'
    if os.path.isfile(filepath):
        mmm_model = utils.load_model(filepath)
    else:
        mmm_model = mmm_run()

    st_obj = MMMStreamlit(mmm_model)
    return st_obj


# Init Streamlit class and plots
st_obj = load_mmm_model()
if 'st_obj' not in st.session_state:
    st.session_state['st_obj'] = st_obj

# Show a view on the train data
st_obj.show_data()
