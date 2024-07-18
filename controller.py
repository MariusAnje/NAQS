import torch

import controller_linear


class Agent():
    def __init__(self, para_space, para_repeat, batch_size=5, lr=0.5,
                 device=torch.device('cpu')):
        self.agent = controller_linear.Agent(
            para_space, para_repeat,
            batch_size=batch_size,
            lr=lr,
            device=device
            )
        self.rollout = self.agent.rollout
        self.store_rollout = self.agent.store_rollout

    def lr_decay(self, lr):
        self.agent.adjust_learning_rate(lr)