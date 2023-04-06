config = {
    'model_name': 'invalid_vs_none_1linear',
    'mode': 'single',
    'load': False,
    'model_to_load': 'xxx',
    'n_episodes': 200000,
    'epsilon': 1,
    'eps_min': 0.01,
    'checkpoint_every': 1000,
    'learn_frequency': 100,
    'learn_amount': 30,
    'gamma': 0.99,
    'learning_rate': 0.0001,
    'memory_size': 50000,
    'batch_size': 32,
    'replace_network_frequency': 1000,
    'eps_dec': 1e-5,
    'invalid_moves_enabled': True,
    'opponent': "NONE",
    'network': "1LINEAR", # 2X8, 4X8, 2X32, 1LINEAR
    'canals': 3,  # 2
    'epsilon_softmax' : False   # true if the epsilon moves should be based on probability
}