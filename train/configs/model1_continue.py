config = {
    'model_name': 'model1_continue',
    'mode': 'single',
    'load': True,
    'model_to_load': 'model1',
    'n_episodes': 400000,
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
    'network': "2X8",
    'canals': 3,
    'epsilon_softmax' : False,
    'adamw_optimizer': False,
    'dropout': False
}