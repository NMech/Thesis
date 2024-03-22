import torch
import numpy       as np
import torch.optim as optim
import xgboost     as xgb
import spinup.algos.pytorch.explainableSac.sacHelpers   as sacHelpers
import spinup.algos.pytorch.explainableSac.core         as core

from copy import deepcopy

def ___Get_actions_from_observations_for_SAC_1D___(sac_policy, observations):

    actions = []
    for observation in observations:
        action = sac_policy(observation, False, False)
        actions.append(action[0].item())

    return actions

def ___Get_actions_from_observations_for_SAC___(sac_policy, observations):
    
    return ___Get_actions_from_observations_for_SAC_1D___(sac_policy, observations)

def ___Get_actions_from_observations_for_GBR___(gbr_policy, observations):

    return gbr_policy.predict(xgb.DMatrix(observations.tolist()))

def ___DGD_calculate_constraint___(sac_policy, gbr_policy, observation_data):
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
    sac_actions = ___Get_actions_from_observations_for_SAC___(sac_policy, observation_data)

    # Calculate actions using GBR policy
    gbr_actions = ___Get_actions_from_observations_for_GBR___(gbr_policy, observation_data)

    sac_actions_array = np.array(sac_actions)
    gbr_actions_array = np.array(gbr_actions.tolist())

    # Compute the mean squared error between actions
    mse = np.mean((sac_actions_array - gbr_actions_array)**2)

    return mse

def ___Compute_loss_pi_for_explainableSac___(updated_sac_policy, alpha, observation_data):
    """
    Compute policy loss for the SAC policy.

    Args:
        updated_sac_policy (object): The updated SAC policy.
        alpha (float): The entropy coefficient.

    Returns:
        tuple: A tuple containing the policy loss and additional information.
     """
    pi, logp_pi = updated_sac_policy(observation_data)
    q1 = core.MLPQFunction(updated_sac_policy.basic_data[0], updated_sac_policy.basic_data[1], updated_sac_policy.basic_data[2], updated_sac_policy.basic_data[3])
    q2 = core.MLPQFunction(updated_sac_policy.basic_data[0], updated_sac_policy.basic_data[1], updated_sac_policy.basic_data[2], updated_sac_policy.basic_data[3])
    q1_pi = q1(observation_data, pi)
    q2_pi = q2(observation_data, pi)
    q_pi = torch.min(q1_pi, q2_pi)

    # Entropy-regularized policy loss
    loss_pi = (alpha * logp_pi - q_pi).mean()

    # Useful info for logging
    pi_info = dict(LogPi=logp_pi.detach().numpy())

    return loss_pi, pi_info

def DGD_calculate_objective(sac_policy, alpha, observation_data):
    """
    Compute the objective function for dual gradient descent (DGD).

    Args:
        sac_policy (callable): The SAC policy.
        alpha (float): The entropy coefficient.
        observation_data (array-like): The observations.

    Returns:
        float: The objective function value.
     """
    loss_pi, _ = ___Compute_loss_pi_for_explainableSac___(sac_policy, alpha, observation_data)  # Compute policy loss

    return -loss_pi

def DGD_calculate_gradient(sac_policy, gbr_policy, observation_data):
    """
     Makes use of the updated sac_policy, which is the one that for a given lambda
     value, gives us the argmin Lagrangian through DGD_find_argmin_lagrangian
      """

    return ___DGD_calculate_constraint___(sac_policy, gbr_policy, observation_data)

def DGD_update_lambda(previous_lambda, step, gradient):
    """
    Update the Lagrange multiplier (lambda) for dual gradient descent (DGD).

    Args:
        previous_lambda (float): The previous lambda value.
        step (float): The step size.
        gradient (float): The gradient value.

    Returns:
        float: The updated lambda value.
     """

    return (previous_lambda + step * gradient)

def DGD_find_argmin_lagrangian(sac_policy, gbr_policy, alpha, lambda_val, observation_data, logger, num_iterations=100, lr=0.001):
    """
    Find the argmin of the Lagrangian using dual gradient descent (DGD).

    Args:
        sac_policy (callable): The SAC policy.
        gbr_policy (callable): The GBR policy.
        alpha (float): The entropy coefficient.
        lambda_val (float): The Lagrange multiplier.
        observation_data (array-like): The observations.
        num_iterations (int): Number of optimization iterations.
        lr (float): Learning rate.

    Returns:
        object: The updated SAC policy.
     """

    logger.log('\n--> DGD_find_argmin_lagrangian start')

    sac_policy_copy = deepcopy(sac_policy)

    # Define optimizer
    optimizer = optim.Adam(sac_policy_copy.parameters(), lr=lr)

    for _ in range(num_iterations):
        # Calculate constraint term using gbr_policy
        constraint_term = ___DGD_calculate_constraint___(sac_policy_copy, gbr_policy, observation_data)

        # Compute loss_pi using sac_policy_copy
        loss_pi, _ = ___Compute_loss_pi_for_explainableSac___(sac_policy, alpha, observation_data)

        # Compute Lagrangian
        lagrangian = -loss_pi + lambda_val * constraint_term

        # Zero gradients, perform backward pass, and update policy parameters
        optimizer.zero_grad()
        lagrangian.backward()
        optimizer.step()

    logger.log('\n--> DGD_find_argmin_lagrangian finish')

    # Return policy parameters at the argmin of the Lagrangian
    return sac_policy_copy

def prepare_data_for_gbr(sac_policy, replay_buffer, logger, replay_buffer_size_to_sample, test_ratio=0.2 ):
    """
    Prepare data from a replay buffer for training a GradientBoostingRegressor (GBR) model.

    Args:
        replay_buffer (ReplayBuffer): The replay buffer containing the samples.
        test_ratio (float):           The ratio of samples to use for testing.

    Returns:
        tuple: A tuple containing X_train, X_test, y_train, and y_test.
    """

    logger.log('\n--> prepare_data_for_gbr start')

    # Sample from the replay buffer
    batch = replay_buffer.sample_batch(int(round(replay_buffer_size_to_sample * replay_buffer.size)))

    # Extract observations and actions from the sampled batch
    obs = batch['obs']

    # Pass observations through the SAC policy to get actions We do that in order to have the 
    # result of the LAST policy. Otherwise when we sample from the replay buffer, we get (obs, action)
    # pairs that come from old policies
    actions = ___Get_actions_from_observations_for_SAC___(sac_policy, obs)

    # Flatten the observations
    obs_flat = obs.reshape((obs.shape[0], -1))

    # Split the data into training and testing sets
    num_samples = obs_flat.shape[0]
    num_test_samples = int(test_ratio * num_samples)

    X_train_tensor = obs_flat[num_test_samples:]
    X_train_list   = X_train_tensor.tolist()
    X_test_tensor  = obs_flat[:num_test_samples]
    X_test_list    = X_test_tensor.tolist()
    y_train_list   = actions[num_test_samples:]
    y_test_list    = actions[:num_test_samples]

    logger.log('\n--> prepare_data_for_gbr finish')

    return X_train_tensor, X_train_list, X_test_tensor, X_test_list, y_train_list, y_test_list

def train_gbr(X_train, y_train, logger, **kwargs):
    """
    Train a Gradient Boosting Regressor (GBR) model using XGBoost.

    Args:
        X_train (array-like): The input features for training.
        y_train (array-like): The target labels for training.
        **kwargs: Additional keyword arguments to pass to the XGBoost model.

    Returns:
        Booster: The trained XGBoost model.
    """

    logger.log('\n--> train_gbr start')

    # Convert data to DMatrix format
    dtrain = xgb.DMatrix(X_train, label=y_train)

    # Train the XGBoost model
    gbr_model = xgb.train(params=kwargs, dtrain=dtrain)

    logger.log('\n--> train_gbr start')

    return gbr_model

def execute_explainable_training(sac_policy, replay_buffer, alpha, logger):

    logger.log('\n--> execute_explainable_training start')

    #########################################################
    # train GBR in order to begin the optimization
    #########################################################
    X_train_tensor, X_train_list, X_test_tensor, X_test_list, y_train_list, y_test_list = prepare_data_for_gbr(sac_policy, replay_buffer, logger, replay_buffer_size_to_sample=0.1, test_ratio=0.2)

    gbr_model = train_gbr(X_train_list, y_train_list, logger)

    #########################################################
    # Initialize lambda
    #########################################################
    lambda_val = 0.1

    #########################################################
    # Find argminL
    #########################################################   
    DGD_find_argmin_lagrangian(sac_policy, gbr_model, alpha, lambda_val, X_train_tensor, logger, num_iterations=100, lr=0.001)

    #########################################################
    # Calculate Gradient
    #########################################################
    gradient = DGD_calculate_gradient(sac_policy, gbr_model, X_train_tensor)

    #########################################################
    # Renew lambda
    #########################################################
    step = 0.01

    logger.log('\n--> Initial Lambda value: \t lambda_val: %f'%lambda_val)
    lambda_val = DGD_update_lambda(lambda_val, step, gradient)
    logger.log('\n--> New Lambda value: \t lambda_val: %f'%lambda_val)

    logger.log('\n--> execute_explainable_training finish')

    # return new_sac_policy, new_gbr_policy