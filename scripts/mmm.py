'''
MMM model class
'''
import logging
import os
from datetime import datetime
from typing import Optional, Union
import pandas as pd
import jax.numpy as jnp
from lightweight_mmm import lightweight_mmm
from lightweight_mmm import utils, preprocessing, plot, optimize_media


class MMMBase:
    '''
    Base MMM class built upon LightweightMMM (https://github.com/google/lightweight_mmm)
    '''

    def __init__(self, target: pd.DataFrame = None, media: pd.DataFrame = None,
                 extra_features: pd.DataFrame = None, costs: pd.DataFrame = None):

        # Check that input data are not missing
        if target is None or target.empty:
            raise ValueError('Missing target data')
        if media is None or media.empty:
            raise ValueError('Missing media data')
        if costs is None or costs.empty:
            raise ValueError('Missing costs')

        # Read data
        self.target = target.copy()
        self.media = media.copy()
        self.costs = costs.copy()

        if extra_features is not None and not extra_features.empty:
            self.extra_features = extra_features.copy()
        else:
            self.extra_features = None

        self.media_vars = self.media.columns
        self.extra_vars = self.extra_features.columns
        self.target_var = 'target'

        # Initialize vars / plots / model
        self.media_scaler = None
        self.extra_features_scaler = None
        self.cost_scaler = None
        self.target_scaler = None
        self.media_scaled = None
        self.extra_features_scaled = None
        self.costs_scaled = None
        self.target_scaled = None
        self.custom_priors = {}
        self.media_effects = None
        self.roi = None
        self.kpi_without_optim = None
        self.starting_allocation = None

        self.opt_time_periods = None
        self.opt_budget = None
        self.opt_prices = None
        self.opt_extra_features = None
        self.optimal_solution = None

        self.fig_media_effects = None
        self.fig_media_posteriors = None
        self.fig_model_fit = None
        self.fig_response_curves = None
        self.fig_roi = None
        self.fig_media_baseline_contribution = None
        self.fig_priors_posteriors = None
        self.fig_pre_post_optim = None

        self.mmm_model = lightweight_mmm.LightweightMMM()

    @staticmethod
    def generate_test_data(data_size: int = 100, n_media_channels: int = 3,
                           n_extra_features: int = 2, geos: int = 1):
        '''
        Generate dummy data for testing/demo
        '''
        # Generate data
        media_data, extra_features, target, costs = utils.simulate_dummy_data(data_size=data_size,
                                                                              n_media_channels=n_media_channels,
                                                                              n_extra_features=n_extra_features,
                                                                              geos=geos)

        # Add to dataframes
        dummy_media = pd.DataFrame(media_data, columns=[
            f'media_{i}' for i in range(n_media_channels)])
        dummy_extra = pd.DataFrame(extra_features, columns=[
            f'extra_{i}' for i in range(n_extra_features)])
        dummy_target = pd.DataFrame(target, columns=['target'])
        dummy_costs = pd.DataFrame(costs).T
        dummy_costs.columns = [f'media_{i}' for i in range(n_media_channels)]

        return dummy_media, dummy_extra, dummy_target, dummy_costs

    @staticmethod
    def train_test_split(target: pd.DataFrame, media_data: pd.DataFrame,
                         extra_features: pd.DataFrame, train_perc: float = 0.9):
        '''
        Split into train and test data
        '''
        data_size = len(target)
        split_point = int(data_size * train_perc)
        media_data_train = media_data.iloc[:split_point, :].copy()
        media_data_test = media_data.iloc[split_point:, :].copy()
        target_train = target.iloc[:split_point].copy()
        target_test = target.iloc[split_point:].copy()
        extra_features_train = extra_features.iloc[:split_point, :].copy()
        extra_features_test = extra_features.iloc[split_point:, :].copy()

        return media_data_train, media_data_test, target_train, target_test, extra_features_train, extra_features_test

    def scale_data(self):
        '''
        Center data around the means
        '''
        # Scale data
        self.media_scaler = preprocessing.CustomScaler(
            divide_operation=jnp.mean)
        self.extra_features_scaler = preprocessing.CustomScaler(
            divide_operation=jnp.mean)
        self.target_scaler = preprocessing.CustomScaler(
            divide_operation=jnp.mean)
        self.cost_scaler = preprocessing.CustomScaler(
            divide_operation=jnp.mean)

        self.media_scaled = self.media_scaler.fit_transform(
            jnp.array(self.media))
        self.extra_features_scaled = self.extra_features_scaler.fit_transform(
            jnp.array(self.extra_features))
        self.target_scaled = self.target_scaler.fit_transform(
            jnp.array(self.target))
        self.costs_scaled = self.cost_scaler.fit_transform(
            jnp.array(self.costs))

    def set_custom_priors(self, priors: Optional[dict] = None):
        '''
        Set up a dict of customer priors
        '''
        if priors is None:
            priors = {"intercept": {'scale': 2}}
        self.custom_priors = {**priors}

    def train(self, n_warmup: int = 1000, n_samples: int = 1000, n_chains: int = 2):
        '''
        Fit/train the model
        '''
        self.scale_data()
        self.mmm_model.fit(media=self.media_scaled,
                           extra_features=self.extra_features_scaled,
                           media_prior=self.costs_scaled.ravel(),
                           target=self.target_scaled.ravel(),
                           number_warmup=n_warmup,
                           number_samples=n_samples,
                           number_chains=n_chains,
                           custom_priors=self.custom_priors)

    def get_diagnostics(self):
        '''
        Plots for diagnostics
        '''
        self.fig_model_fit = plot.plot_model_fit(
            media_mix_model=self.mmm_model, target_scaler=self.target_scaler)
        self.fig_priors_posteriors = plot.plot_prior_and_posterior(
            media_mix_model=self.mmm_model)
        self.fig_media_posteriors = plot.plot_media_channel_posteriors(media_mix_model=self.mmm_model,
                                                                       channel_names=self.media_vars)
        self.fig_response_curves = plot.plot_response_curves(media_mix_model=self.mmm_model,
                                                             media_scaler=self.media_scaler,
                                                             target_scaler=self.target_scaler)

    def plot_media_effects(self):
        '''
        Plots of media effects and ROIs
        '''
        self.media_effects, self.roi = self.mmm_model.get_posterior_metrics(unscaled_costs=self.costs.to_numpy().ravel(),
                                                                            target_scaler=self.target_scaler,
                                                                            cost_scaler=self.cost_scaler)
        self.fig_media_baseline_contribution = plot.plot_media_baseline_contribution_area_plot(media_mix_model=self.mmm_model,
                                                                                               target_scaler=self.target_scaler,
                                                                                               channel_names=self.media_vars)
        self.fig_media_effects = plot.plot_bars_media_metrics(
            metric=self.media_effects, channel_names=self.media_vars)
        self.fig_roi = plot.plot_bars_media_metrics(
            metric=self.roi, channel_names=self.media_vars)

    def run_optimization(self,
                         n_time_periods: int = None,
                         budget: Union[float, int] = None,
                         prices: list = None,
                         extra_features_opt: Optional[pd.DataFrame] = None):
        '''
        Run media optimization.
        '''
        if extra_features_opt is not None and not extra_features_opt.empty:
            extra_features_opt_array = self.extra_features_scaler.transform(
                jnp.array(extra_features_opt.iloc[:n_time_periods, :]))
        else:
            extra_features_opt_array = None

        self.opt_time_periods = n_time_periods
        self.opt_budget = budget
        self.opt_prices = prices
        self.opt_extra_features = extra_features_opt

        self.optimal_solution, self.kpi_without_optim, self.starting_allocation = \
            optimize_media.find_optimal_budgets(
                n_time_periods=n_time_periods,
                media_mix_model=self.mmm_model,
                budget=budget,
                extra_features=extra_features_opt_array,
                prices=jnp.array(prices),
                target_scaler=self.target_scaler,
                media_scaler=self.media_scaler)

        self.fig_pre_post_optim = plot.plot_pre_post_budget_allocation_comparison(media_mix_model=self.mmm_model,
                                                                                  kpi_with_optim=self.optimal_solution[
                                                                                      'fun'],
                                                                                  kpi_without_optim=self.kpi_without_optim,
                                                                                  optimal_buget_allocation=self.optimal_solution[
                                                                                      'x'],
                                                                                  previous_budget_allocation=self.starting_allocation,
                                                                                  channel_names=self.media_vars)


def run():
    '''
    Run the model
    '''
    current_time = datetime.now()
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        filename=f'logs/log_{current_time}.log', level=logging.INFO)

    logging.info('Starting')
    logging.info('Generating dummy data')
    df_media, df_extra, df_target, df_costs = MMMBase.generate_test_data()
    df_media_train, df_media_test, df_target_train, df_target_test, df_extra_train, df_extra_test = \
        MMMBase.train_test_split(
            media_data=df_media, target=df_target, extra_features=df_extra, train_perc=0.9)

    logging.info('Running model')
    mmm_model = MMMBase(target=df_target_train, media=df_media_train,
                        extra_features=df_extra_train, costs=df_costs)
    mmm_model.set_custom_priors()
    mmm_model.train(n_warmup=500, n_samples=500, n_chains=1)

    mmm_model.get_diagnostics()
    mmm_model.plot_media_effects()

    logging.info('Running optimization')
    mmm_model.run_optimization(
        n_time_periods=len(df_extra_test), budget=60, prices=[0.1, 0.11, 0.12], extra_features_opt=df_extra_test)

    logging.info('Saving')
    os.chdir('./')
    os.makedirs('results', exist_ok=True)
    utils.save_model(mmm_model, 'results/model.pkl')

    logging.info('Finished')

    return mmm_model
