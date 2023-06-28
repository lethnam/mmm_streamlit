'''
Streamlit class to load MMM data and create visuals
'''
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from lightweight_mmm import utils


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

    def load_model(self, folderpath: str = './results/', filename: str = 'model.pkl'):
        '''
        Load MMM results
        '''
        if os.path.isfile(folderpath+filename):
            mmm_model = utils.load_model(folderpath+filename)
        else:
            raise ValueError('The requested model does not exist')

        return mmm_model

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
        fig_heatmap, ax_heatmap = plt.subplots(figsize=(10, 10))
        sns.heatmap(ax=ax_heatmap, data=self.df_data.corr(),
                    annot=True, fmt='.2f')
        st.pyplot(fig=fig_heatmap, use_container_width=True)

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

    def show_diagnostics(self):
        '''
        Display posterior summary table, and model fit
        '''
        st.subheader('Summary')
        st.write(
            f"Divergence: {int(self.mmm_model._divergence) if self.mmm_model._divergence is not None else 'None'}")
        st.pyplot(self.mmm_model.fig_diagnostics)

        st.subheader('Model fit')
        st.pyplot(self.mmm_model.fig_model_fit)

        st.subheader('Priors vs. Posteriors')
        st.pyplot(self.mmm_model.fig_priors_posteriors)

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
        st.subheader(
            f'Optimal allocation on a budget of {self.mmm_model.opt_budget}')
        st.pyplot(self.mmm_model.fig_pre_post_optim)
