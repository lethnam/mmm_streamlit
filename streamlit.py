'''
Visualize with streamlit
'''
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import mmm
import streamlit as st
from lightweight_mmm import utils


class MMMStreamlit:
    '''
    Streamlit class to display MMM results
    '''

    def __init__(self, mmm_model):
        st.header('Loading MMM results ...')
        self.mmm_model = mmm_model

        self.df_target = self.mmm_model.target
        self.df_media = self.mmm_model.media
        self.df_extra_features = self.mmm_model.extra_features
        self.df_data = pd.concat(
            [self.df_target, self.df_media, self.df_extra_features], axis=0)
        self.media_vars = self.df_media.columns

    def show_data(self):
        '''
        Display train data
        '''
        st.header('Train data set')
        st.dataframe(self.df_data.head(20))

    def eda_plots(self):
        '''
        Plots EDA on train data
        '''
        st.header('EDA on train data')
        st.subheader('Correlation heatmap')
        fig_heatmap, ax_heatmap = plt.subplot()
        sns.heatmap(ax=ax_heatmap, data=self.df_data.corr())
        st.pyplot(fig=fig_heatmap)

        st.subheader('Target vs. Media')
        st.line_chart(data=self.df_data[['target']+self.media_vars])


# Load MMM results. Run the model if there is no result file
filepath = 'results/model.pkl'
if os.path.isfile(filepath):
    mmm_model = utils.load_model(filepath)
else:
    mmm_model = mmm.run()

# Init Streamlit class and plots
st_obj = MMMStreamlit(mmm_model)
st_obj.show_data()
st_obj.eda_plots()
