config = {
    'model_name': 'invalid_vs_rl_continue',
    'mode': 'single',   # single, competitive, cooperative, single_lookback
    'load': True,      # true if loading a previously trained model
    'model_to_load': 'invalid_vs_rl',
    'n_episodes': 100000,        #default is 5000 for invalid_moves_disabled
    'epsilon': 0.01,       # 0.01 if loading a model (usually), 0.5is default for invalid_moves_disabled
    'eps_min': 0.01,
    'checkpoint_every': 1000,   # default is 100 for invalid_moves_disabled
    'learn_frequency': 100,     # default is 100 for invalid_moves_disabled
    'learn_amount': 30,        # default is 100 for invalid moves_disabled
    'gamma': 0.99,
    'learning_rate': 0.0001,
    'memory_size': 50000,
    'batch_size': 32,
    'replace_network_frequency': 1000,  # default is 100 for invalid moves_disabled
    'eps_dec': 1e-5,
    'invalid_moves_enabled': True,
    'opponent': 'RL', # NONE, RANDOM, MINMAX OR RL
}