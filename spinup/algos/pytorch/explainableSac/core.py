import numpy as np
import scipy.signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


LOG_STD_MAX = 2
LOG_STD_MIN = -20

class SquashedGaussianMLPActor(nn.Module):

# This class represents the actor network in the Soft Actor-Critic (SAC) algorithm.
# It is a feedforward neural network (Multi-Layer Perceptron, MLP) that takes observations as input and outputs actions.
# The actions are sampled from a Gaussian distribution parameterized by the mean (mu) and standard deviation (log_std) predicted by the network.
# The Gaussian distribution is then squashed to ensure that the actions are within the valid action range of the environment.
# This class also computes the log probability of the sampled actions under the Gaussian distribution.

# obs_dim: Integer, the dimensionality of the observation space.
# act_dim: Integer, the dimensionality of the action space.
# hidden_sizes: Tuple of integers, specifying the sizes of hidden layers in the neural network.
# activation: Activation function, such as ReLU or Tanh, used in the hidden layers.
# act_limit: Float, the limit or range of valid actions in the environment.

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding 
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290) 
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi


class MLPQFunction(nn.Module):

# This class represents the Q-function network in the SAC algorithm.
# It is a feedforward neural network (MLP) that takes both observations and actions as input and outputs Q-values.
# The Q-values represent the expected return (cumulative rewards) when taking a specific action in a given state.
# The Q-function is used to estimate the value of state-action pairs and is essential for policy improvement in reinforcement learning algorithms like SAC.

# obs_dim: Integer, the dimensionality of the observation space.
# act_dim: Integer, the dimensionality of the action space.
# hidden_sizes: Tuple of integers, specifying the sizes of hidden layers in the neural network.
# activation: Activation function, such as ReLU or Tanh, used in the hidden layers.

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class MLPActorCritic(nn.Module):

# This class combines the actor and critic networks into a single model for training in the SAC algorithm.
# It contains instances of both the SquashedGaussianMLPActor and MLPQFunction classes.
# The SquashedGaussianMLPActor is responsible for learning the policy (actor), while the MLPQFunction instances represent the value functions (critics).
# This class provides methods for acting in the environment (act method) and for computing Q-values given states and actions. It also initializes the actor and critic networks.

# observation_space: Object, the observation space of the environment.
# action_space: Object, the action space of the environment.
# hidden_sizes: Tuple of integers, specifying the sizes of hidden layers in both actor and critic neural networks.
# activation: Activation function, such as ReLU or Tanh, used in the hidden layers.

    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a.numpy()
