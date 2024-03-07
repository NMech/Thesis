import numpy       as np
import torch.optim as optim

def DGD_calculate_constraint(sac_policy, gbr_policy, observation_data):
    """
    Calculate the dual gradient descent (DGD) constraint between SAC policy and GBR policy.

    Args:
        sac_policy (function): The SAC policy (actor) function that maps observations to actions.
        gbr_policy (function): The GBR policy function that maps observations to actions.
        observation_data (array-like): The observations for which to compute the constraint.

    Returns:
        float: The value of the DGD constraint.
    """

    # Calculate actions using SAC policy
    sac_actions = sac_policy(observation_data)

    # Calculate actions using GBR policy
    gbr_actions = gbr_policy(observation_data)

    sac_actions_array = np.array(actions1)
    gbr_actions_array = np.array(actions2)

    # Compute the mean squared error between actions
    mse = np.mean((sac_actions_array - gbr_actions_array)**2)

    return mse

def DGD_calculate_objective(observation_data):

    loss_pi, _ = compute_loss_pi(observation_data)  # Compute policy loss

    return -loss_pi

def DGD_find_argmin_lagrangian(sac_policy, gbr_policy, lambda_val, num_iterations=100, lr=0.001):
    # Define optimizer
    optimizer = optim.Adam(sac_policy.parameters(), lr=lr)

    for _ in range(num_iterations):
        # Calculate constraint term using gbr_policy
        constraint_term = calculate_distance(sac_policy, gbr_policy)

        # Compute loss_pi using sac_policy
        loss_pi, _ = compute_loss_pi(data)

        # Compute Lagrangian
        lagrangian = -loss_pi + lambda_val * constraint_term

        # Zero gradients, perform backward pass, and update policy parameters
        optimizer.zero_grad()
        lagrangian.backward()
        optimizer.step()

    # Return policy parameters at the argmin of the Lagrangian
    return sac_policy.parameters()

def DGD_calculate_gradient(sac_policy, gbr_policy, observation_data):
     """
     Makes use of the updated sac_policy, which is the one that for a given lamda
     value, gives us the argmin Lagrangian through DGD_find_argmin_lagrangian
      """

    return DGD_calculate_constraint(sac_policy, gbr_policy, observation_data)

def DGD_update_lamda(previous_lamda, step, gradient):

    return (lamda + step * gradient)

def train_GBM():

    pass


