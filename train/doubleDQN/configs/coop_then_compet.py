config = {
    'model_name': '200k-coop_200k-compet',
    'mode': 'competitive',
    'load': True,
    'model_to_load': '200k-coop',
    'n_episodes': 200000,
    'epsilon': 0.01,
    'eps_min': 0.01,
    'checkpoint_every': 1000,
    'learn_frequency': 100,
    'learn_amount': 30,
    'reward_for_win': 100,
    'gamma': 0.99,
    'learning_rate': 0.0001,
    'memory_size': 50000,
    'batch_size': 32,
    'replace_network_frequency': 10000,
    'eps_dec': 1e-5,
    'invalid_moves_enabled': True,
}