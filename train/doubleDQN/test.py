from dueling_dqn_agent import DuelingDQNAgent
from myenv import MyEnv
from configs import invalid_vs_none_then_minmax as conf

C = conf.config
n_episodes = C.get('n_episodes')
epsilon = C.get('epsilon')
eps_min = C.get('eps_min')
checkpoint_every = C.get('checkpoint_every')
learn_frequency = C.get('learn_frequency')
learn_amount = C.get('learn_amount')
mode = C.get('mode')
gamma = C.get('gamma')
lr = C.get('learning_rate')
mem_size = C.get('memory_size')
batch_size = C.get('batch_size')
replace = C.get('replace_network_frequency')
eps_dec = C.get('eps_dec')
invalid_moves_enabled = C.get('invalid_moves_enabled')
opponent = C.get('opponent')
seed = C.get('model_name')
# seed = "61603"
# seed = "test-only-valid-moves"

invalid_moves_enabled = False

def main():
    env = MyEnv(mode, seed, opponent, checkpoint_every)
    env.reset()
    agent = DuelingDQNAgent(gamma=gamma, epsilon=0, lr=lr,
                            input_dims=env.observation_space.shape,
                            n_actions=env.action_space.n, mem_size=mem_size, eps_min=eps_min,
                            batch_size=batch_size, replace=replace, eps_dec=eps_dec,
                            learn_amount=learn_amount, seed=seed, checkpoint_dir='models/',
                            invalid_moves_enabled=invalid_moves_enabled)

    agent.load_models(seed)

    done = False
    observation = env.reset()
    print("start: " + " seed: " + str(seed) + " timestamp: " + " mode: " + str(mode) + "\n")
    print(env.render())

    n_steps = 0
    score = 0
    while not done:
        action = agent.choose_action(observation,env)
        observation_, reward, done, info = env.primary_player_step(action)
        action_log = "------------player: " + info.get("player") + " move: " + info.get(
            "move") + " which is: " + info.get("valid") + ": " + info.get("message") + " reward: " + str(
            reward) + " score: " + str(score) + "\n"
        print(action_log)
        print(env.render())
        n_steps += 1

        if not done:
            observation_, reward, done, info = env.secondary_player_step()
            action_log = "------------player: " + info.get("player") + " move: " + info.get(
                "move") + " which is: " + info.get("valid") + ": " + info.get("message") + " reward: " + str(
                reward) + " score: " + str(score) + "\n"
            print(action_log)
            print(env.render())
            n_steps += 1

        observation = observation_
        reward = env.calculate_reward()
        score += reward

    episode_log  = 'end: ' + " seed: " + str(seed) +  ' score: ' +str(score)  + ' steps ' + str(n_steps) + "\n"
    print(episode_log)

main()