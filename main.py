import argparse
import csv
import logging
import os
import time

import torch

from controller import Agent
from config import ARCH_SPACE, QUAN_SPACE, CLOCK_FREQUENCY, NEW_CHOICE
from utility import BestSamples
import utility
import numpy as np
from tqdm import tqdm


# def get_args():
parser = argparse.ArgumentParser('Parser User Input Arguments')
parser.add_argument(
    '-l', '--layers',
    type=int,
    default=6,
    help="the number of child network layers, default is 6"
    )
parser.add_argument(
    '-ep', '--episodes',
    type=int,
    default=200,
    help='''the number of episodes for training the policy network, default
        is 2000'''
    )
parser.add_argument(
    '-e', '--epochs',
    type=int,
    default=100,
    help='''the number of epochs for training each DNN model, default
        is 100'''
    )
parser.add_argument(
    '-lr', '--learning_rate',
    type=float,
    default=0.2,
    help="learning rate for updating the controller, default is 0.2")
parser.add_argument(
    '-s', '--seed',
    type=int,
    default=1,
    help="seed for randomness, default is 0"
    )
parser.add_argument(
    '-g', '--gpu',
    type=int,
    default=0,
    help="in single gpu mode the id of the gpu used, default is 0"
    )
parser.add_argument(
    '-k', '--skip',
    action='store_true',
    help="include skip connection in the architecture, default is false"
    )
parser.add_argument(
    '-m', '--multi-gpu',
    action='store_true',
    help="use all gpus available, default false"
    )
parser.add_argument(
    '-v', '--verbosity',
    type=int,
    choices=range(3),
    default=0,
    help="verbosity level: 0 (default), 1 and 2 with 2 being the most verbose"
    )
parser.add_argument(
    '-n', '--nodes',
    type=int,
    default=10,
    help="number of nodes"
    )
args = parser.parse_args()


def get_logger(filepath=None):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)
    if filepath is not None:
        file_handler = logging.FileHandler(filepath+'.log', mode='w')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(file_handler)
    return logger


def main():
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available()
                          else "cpu")
    print(f"using device {device}")
    dir = os.path.join(
        f'experiment',
        f"{args.layers} layers"
        )
    if os.path.exists(dir) is False:
        os.makedirs(dir)
    nas(device, dir)


def nas(device, dir='experiment'):
    filepath = os.path.join(dir, f"{time.time()}")
    logger = get_logger(filepath)
    logger.info(f"INFORMATION")
    logger.info(f"skip connection: \t\t\t {args.skip}")
    logger.info(f"controller learning rate: \t\t {args.learning_rate}")
    logger.info(f"architecture episodes: \t\t\t {args.episodes}")
    logger.info(f"using multi gpus: \t\t\t {args.multi_gpu}")
    logger.info(f"number of nodes: \t\t\t {args.nodes}")
    logger.info(f"number of layers: \t\t\t {args.layers}")
    logger.info(f"architecture space: ")

    NODE_SPACE = NEW_CHOICE
    agent = Agent(NODE_SPACE, args.layers,
                  lr=args.learning_rate,
                  device=torch.device('cpu'))

    arch_id, total_time = 0, 0
    logger.info('=' * 50 + "Start exploring architecture space" + '=' * 50)
    logger.info('-' * len("Start exploring architecture space"))

    best_samples = BestSamples(5)
    # for e in tqdm(range(args.episodes)):
    for e in range(args.episodes):
        arch_id += 1
        start = time.time()
        arch_rollout, arch_paras = agent.rollout()

        # model = create_model_from_rollout(arch_rollout[0:])
        # train_model(args.epochs)
        # multiplier = create_multiplier_from_rollout(arch_rollout[0])
        # accuracy = get_accuracy(model, multiplier)
        # latency, energy, power = get_hardware(model, multipler)
        # arch_reward = alpha * accuracy - beta * latency - gamma * energy - theta * power

        arch_reward = np.sum(arch_rollout) # something need to be replaced, this one just lets the code run
        agent.store_rollout(arch_rollout, arch_reward)
        end = time.time()
        ep_time = end - start
        total_time += ep_time
        best_samples.register(arch_id, arch_rollout, arch_reward)
        logger.info(f"Architecture Reward: {arch_reward}, " +
                    f"Elasped time: {ep_time}, " +
                    f"Average time: {total_time/(e+1)}")
        b_reward = best_samples.best_reward()
        logger.info(f"Best Reward: {b_reward[0]}, " +
                    f"ID: {b_reward[1]}, " +
                    f"Rollout: {b_reward[2]}")
        logger.info('-' * len("Start exploring architecture space"))
    logger.info(
        '=' * 50 + "Architecture sapce exploration finished" + '=' * 50)
    logger.info(f"Total elasped time: {total_time}")
    logger.info(f"Best samples: {best_samples}")

    b_reward = best_samples.best_reward()
    arch_rollout = b_reward[2]
    # print(arch_rollout)

if __name__ == '__main__':
    import random
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    main()

