"""
Agent base class to work with the PFRL library
"""

import numpy as np
import torch
from constants import MODEL_PATH


class AgentBase:
    def __init__(self,
                 env,
                 agent,
                 episodes,
                 max_episode_len,
                 exp_id,
                 print_freq=500,
                 save_freq=10000,
                 writer=None,
                 threshold=0.8
                ):
        """

        Args:
            env (gym obj): Instance of initialized env
            agent (pfrl obj): Instantiated pfrl agent
            episodes (int): Num of episodes
            max_episode_len (int): Per episode len
            exp_id (str): ID of experiment (used as unique id for saving agents)
            print_freq (int): Number of steps after which to print info
            save_freq (int): Number of steps after which to save agent's internal state
            writer (torch.utils.tensorboard.SummaryWriter): Tensorboard logging object
            threshold (int): If stock mat prob > threshold, then action will be taken on that
            stock
        """
        self.env = env
        self.agent = agent
        self.episodes = episodes
        self.max_episode_len = max_episode_len
        self.print_freq = print_freq
        self.save_freq = save_freq
        self.writer = writer
        self.exp_id = exp_id
        self.threshold = threshold

    def fit(self):
        for i in range(1, self.episodes + 1):
            mean_reward = 0  # return (sum of rewards)
            steps = 0  # time step
            state = self.env.get_state_info()
            checks = False

            while True:
                action = self.agent.act(state)
                stock_mat = self.agent.model.forward_2(state)
                # stock_mat, action = self.custom_act(state)
                # print(stock_mat)
                stock_mat = (stock_mat >= self.threshold).astype(int)
                # random_indices = np.random.choice(np.arange(430), 5,
                #                                   replace=False)
                # stock_mat = np.zeros((num_stocks))
                # stock_mat[random_indices] = 1
                obs, reward = self.env.step(action, stock_mat)
                mean_reward += reward
                steps += 1

                reset = steps == self.max_episode_len

                self.agent.observe(obs, reward, False, reset)

                if reset:
                    break
                if steps % self.print_freq == 0:
                    print('Total Steps: {} Reward {}'.format(steps, mean_reward))
                    stats = self.agent.get_statistics()
                    print(stats)
                    if not checks:
                        assert stats[1][0] == 'average_loss'
                        checks = True

                    avg_loss = stats[1][1]
                    print("Stock Mat")
                    print(stock_mat)
                    print("*"*10)
                    if self.writer:
                        # Mean reward is per print_freq steps
                        self.writer.add_scalar('Rewards/Mean Reward', mean_reward, steps)
                        self.writer.add_scalar('Loss/Avg Loss', avg_loss, steps)

                    mean_reward = 0

                if steps % self.save_freq == 0:
                    print('Agent saved at step: {}'.format(steps))
                    agent_name = "{}_{}".format(self.exp_id, steps)
                    self.agent.save(str(MODEL_PATH / 'saved_agents' / agent_name))

        print('Finished.')
        if self.writer:
            self.writer.flush()
            self.writer.close()







