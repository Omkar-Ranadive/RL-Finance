"""
Agent base class to work with the PFRL library
"""

import numpy as np


class AgentBase:
    def __init__(self,
                 env,
                 agent,
                 episodes,
                 max_episode_len,
                 print_freq=500,
                 save_freq=10000,
                 writer=None,
                ):
        """

        Args:
            env (gym obj): Instance of initialized env
            agent (pfrl obj): Instantiated pfrl agent
            episodes (int): Num of episodes
            max_episode_len (int): Per episode len
            writer (torch.utils.tensorboard.SummaryWriter): Tensorboard logging object
        """
        self.env = env
        self.agent = agent
        self.episodes = episodes
        self.max_episode_len = max_episode_len
        self.print_freq = print_freq
        self.save_freq = save_freq
        self.writer = writer

    def fit(self):
        for i in range(1, self.episodes + 1):
            R = 0  # return (sum of rewards)
            steps = 0  # time step
            state = self.env.get_state_info()
            num_stocks = state[0].shape[0]
            checks = False

            while True:
                action = self.agent.act(state)
                random_indices = np.random.choice(np.arange(430), 5,
                                                  replace=False)
                stock_mat = np.zeros((num_stocks))
                stock_mat[random_indices] = 1

                obs, reward = self.env.step(action, stock_mat)
                R += reward
                steps += 1
                reset = steps == self.max_episode_len

                self.agent.observe(obs, reward, False, reset)
                if reset:
                    break
                if steps % self.print_freq == 0:
                    print('Total Steps: {} Reward {}'.format(steps, reward))
                    stats = self.agent.get_statistics()
                    print(stats)
                    if not checks:
                        assert stats[1][0] == 'average_loss'
                        checks = True

                    avg_loss = stats[1][1]

                    if self.writer:
                        self.writer.add_scalar('Rewards/Total Reward', R, steps)

                        self.writer.add_scalar('Loss/Avg Loss', avg_loss, steps)

        print('Finished.')
        if self.writer:
            self.writer.flush()
            self.writer.close()










