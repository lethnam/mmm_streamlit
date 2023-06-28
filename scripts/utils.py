import os
import logging
from operator import attrgetter
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lightweight_mmm import utils as mmm_utils
from numpyro.diagnostics import summary as numpyro_summary
from scripts.mmm import MMMBase


def generate_demo_model(train_split_perc: float = 0.9):
    '''
    Generate a demo model
    '''
    current_time = datetime.now()
    os.makedirs('./logs', exist_ok=True)
    logging.basicConfig(
        filename=f'./logs/log_{current_time}.log', level=logging.INFO)

    logging.info('Starting')
    logging.info('Generating dummy data')
    df_media, df_extra, df_target, df_costs = MMMBase.generate_test_data()
    df_media_train, df_media_test, df_target_train, df_target_test, df_extra_train, df_extra_test = \
        MMMBase.train_test_split(
            media_data=df_media, target=df_target, extra_features=df_extra, train_perc=train_split_perc)

    logging.info('Running model')
    mmm_model = MMMBase(target=df_target_train, media=df_media_train,
                        costs=df_costs, extra_features=df_extra_train,
                        target_test=df_target_test, media_test=df_media_test,
                        extra_features_test=df_extra_test)
    mmm_model.set_custom_priors()
    mmm_model.train(n_warmup=500, n_samples=500, n_chains=1)

    mmm_model.get_diagnostics()
    mmm_model.plot_media_effects()

    return mmm_model


def save_model(mmm_model, folderpath: str = 'results/', filename: str = 'model.pkl'):
    '''
    Save trained model object
    '''
    os.makedirs(folderpath, exist_ok=True)
    mmm_utils.save_model(mmm_model, folderpath+filename)


def mcmc_diagnostics(mcmc):
    '''
    Retrieve diagnostics from numpyro mcmc object
    '''
    divergence = np.sum(mcmc.get_extra_fields()[
                        'diverging']) if 'diverging' in mcmc.get_extra_fields() else None
    state_sample_field = attrgetter(mcmc._sample_field)(mcmc._last_state)
    sample_summary = numpyro_summary(mcmc._states['z'])
    n_eff = np.concatenate([sample_summary[i]['n_eff'].ravel()
                           for i in state_sample_field])
    r_hat = np.concatenate([sample_summary[i]['r_hat'].ravel()
                           for i in state_sample_field])

    fig_diag, axes = plt.subplots(1, 2, figsize=(15, 10))
    sns.histplot(data=n_eff, ax=axes[0])
    sns.histplot(data=r_hat, ax=axes[1])
    axes[0].set_title('Effective sample sizes')
    axes[1].set_title('r_hat')

    return divergence, n_eff, r_hat, fig_diag
