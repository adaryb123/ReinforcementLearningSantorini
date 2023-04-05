config = {
    'model_name': 'valid_vs_none',
    'mode': 'single',   # single, competitive, cooperative, single_lookback
    'load': False,      # true if loading a previously trained model
    'model_to_load': 'invalid_vs_none_2x32',
     'n_episodes': 5000,        #default is 5000 for invalid_moves_disabled
    'epsilon': 0.5,       # 0.01 if loading a model (usually), 0.5is default for invalid_moves_disabled
    'eps_min': 0.01,
    'checkpoint_every': 100,   # default is 100 for invalid_moves_disabled
    'learn_frequency': 100,     # default is 100 for invalid moves_disabled
    'learn_amount': 100,         # default is 100 for invalid moves_disabled
    'gamma': 0.99,
    'learning_rate': 0.0001,
    'memory_size': 50000,
    'batch_size': 32,
    'replace_network_frequency': 100,  # default is 100 for invalid moves_disabled
    'eps_dec': 1e-5,
    'invalid_moves_enabled': False,
    'opponent': 'RL', # NONE, RANDOM, MINMAX OR RL
    'network': '2X8',  # 2X8, 4X8, 2X32, 1LINEAR
    'canals': 3 # 2
}