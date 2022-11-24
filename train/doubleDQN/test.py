from dueling_dqn_agent import DuelingDQNAgent
from myenv import MyEnv

# ENTER A SEED OF A PRE TRAINED MODEL
seed = 3

def main():
    env = MyEnv()
    env.reset()
    agent = DuelingDQNAgent(gamma=0.99, epsilon=0, lr=0.0001,
                            input_dims=(env.observation_space.shape),
                            n_actions=env.action_space.n, mem_size=50000, eps_min=0,
                            batch_size=32, replace=10000, eps_dec=0,
                            chkpt_dir='models/', algo='DuelingDQNAgent_' + str(seed),
                            env_name='Santorini')

    agent.load_models()

    done = False
    observation = env.reset()
    print("start test \n")
    print(env.render())

    n_steps = 0
    score = 0
    while not done:
        action = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        score += reward
        action_log = "------------player: " + info.get("player") + " move: " + info.get("move") + " which is: " + info.get("valid") + ": " + info.get("message") + "\n"
        print(action_log)
        print(env.render())

        observation = observation_
        n_steps += 1

    episode_log  = 'end test: ' + ' score: ' +str(score)  + ' steps ' + str(n_steps) + "\n"
    print(episode_log)

main()