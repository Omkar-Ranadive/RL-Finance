"""
Agent base class to work with the PFRL library
"""

import numpy as np
import torch
from constants import MODEL_PATH
import logging


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

        #TODO - Logging won't work due to the secondary thread not terminating in PCAP Reader
        # For now, save normally by opening a file

        # Initialize logger to save information to file
        self.fname = str(MODEL_PATH / 'runs' / '{}.log'.format(self.exp_id))
        # logging.basicConfig(filename='test.log', level=logging.INFO, format='%(message)s')

    def fit(self):
        with open(self.fname, 'w') as f:
            for i in range(1, self.episodes + 1):
                mean_reward = 0  # return (sum of rewards)
                hits = 0  # To calculate hit ratio
                steps = 0  # time step
                state = self.env.get_state_info()
                checks = False
                returns = []   # Keep track of returns to calculate standard deviation
                # Keep track of actions taken
                action_count = [0 for _ in range(self.env.action_space_d.n)]

                while True:
                    action = self.agent.act(state)
                    stock_mat_prob = self.agent.model.forward_2(state)
                    mean_prob = np.mean(stock_mat_prob)
                    cur_threshold = min(self.threshold, mean_prob)
                    stock_mat = (stock_mat_prob >= cur_threshold).astype(int)

                    obs, reward = self.env.step(action, stock_mat)
                    mean_reward += reward
                    steps += 1

                    # Increase action count
                    action_count[action] += 1

                    # Keep track of variables required to calculate Sharpe Ratio (only if action =  sell)
                    if action == 1:
                        roi = self.env.get_sell_info()
                        returns.append(roi)

                    # If reward > 0, then it is a profitable trade
                    if reward > 0:
                        hits += 1

                    reset = steps == self.max_episode_len

                    self.agent.observe(obs, reward, False, reset)

                    if reset:
                        break
                    if steps % self.print_freq == 0:
                        hit_ratio = hits/self.print_freq   # Num of trade executed = print freq

                        # A ratio of 0 indicates, no selling of equities happened
                        sharpe_ratio = 0
                        mean_roi = 0
                        std_roi = 0

                        if len(returns) > 0:
                            mean_roi = np.mean(returns)
                            std_roi = np.std(returns)
                            sharpe_ratio = (mean_roi - 0.01) / std_roi

                        metric_info = 'Total Steps: {} Reward {} Hit Ratio {} Sharpe Ratio {} ' \
                                      'Mean ROI {} STD ROI {}'.format(steps, mean_reward, hit_ratio,
                                                                      sharpe_ratio, mean_roi, std_roi)
                        action_info = 'Buy Count: {} Sell Count: {} Hold Count: {}'.format(
                            action_count[0], action_count[1], action_count[2])

                        print(metric_info)

                        #TODO Logging won't work due to non-terminating pcap thread
                        # logging.info(metric_info)
                        # logging.info(action_info)
                        f.write("*"*10 + '\n')
                        f.write("Iteration: {}\n".format(steps))
                        f.write(metric_info + '\n')
                        f.write(action_info + '\n')

                        stats = self.agent.get_statistics()
                        print(stats)
                        if not checks:
                            assert stats[1][0] == 'average_loss'
                            checks = True

                        avg_loss = stats[1][1]
                        print("Stock Mat Prob: ")
                        print(stock_mat_prob)
                        print("*"*10)
                        print("Stock Mat")
                        print(stock_mat)
                        print("*"*10)

                        if self.writer:
                            # Mean reward is per print_freq steps
                            self.writer.add_scalar('Metrics/Mean Reward', mean_reward, steps)
                            self.writer.add_scalar('Metrics/Hit Ratio', hit_ratio, steps)
                            self.writer.add_scalar('Metrics/Sharpe Ratio', sharpe_ratio, steps)
                            self.writer.add_scalar('Loss/Avg Loss', avg_loss, steps)

                        # Reset the metrics
                        mean_reward = 0
                        hits = 0
                        returns = []
                        action_count = [0 for _ in range(self.env.action_space_d.n)]

                    if steps % self.save_freq == 0:
                        print('Agent saved at step: {}'.format(steps))
                        agent_name = "{}_{}".format(self.exp_id, steps)
                        self.agent.save(str(MODEL_PATH / 'saved_agents' / agent_name))

            print('Finished.')
            f.close()
            if self.writer:
                self.writer.flush()
                self.writer.close()







