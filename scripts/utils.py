import os
import logging
from datetime import datetime
from lightweight_mmm import utils as mmm_utils
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
