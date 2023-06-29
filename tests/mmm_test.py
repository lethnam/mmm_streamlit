from scripts.mmm import MMMBase

# TODO: add proper tests


def test_run():
    '''
    Test if the class is working
    '''
    df_media, df_extra, df_target, df_costs = MMMBase.generate_test_data(
        100, 3, 2, 1)
    assert df_media.shape[1] == 3
    assert df_extra.shape[1] == 2
    assert len(df_target) == 100
    assert df_costs.shape[1] == 3

    mmm_model = MMMBase(target=df_target, media=df_media,
                        costs=df_costs, extra_features=df_extra)
    mmm_model.train(n_warmup=100, n_samples=100, n_chains=1)
    assert isinstance(mmm_model, MMMBase)
    assert hasattr(mmm_model.mmm_model, '_mcmc')
