config = {
    'model_name': 'invalid_vs_none_2x8_then_heuristic_2x',
    'mode': 'single',
    'load': True,
    'model_to_load': 'invalid_vs_none_2x8_then_heuristic',
    'n_episodes': 100000,
    'epsilon': 0.01,
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
    'opponent': "HEURISTIC",
    'network': "2X8", # 2X8, 4X8, 2X32, 1LINEAR
    'canals': 3,  # 2
    'epsilon_softmax' : False   # true if the epsilon moves should be based on probability
}