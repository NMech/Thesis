import numpy       as np
import torch.optim as optim
import xgboost     as xgb

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

    sac_actions_array = np.array(sac_actions)
    gbr_actions_array = np.array(gbr_actions)

    # Compute the mean squared error between actions
    mse = np.mean((sac_actions_array - gbr_actions_array)**2)

    return mse

def compute_loss_pi_for_explainableSac(updated_sac_policy, alpha, data):
    """
    Compute policy loss for the SAC policy.

    Args:
        updated_sac_policy (object): The updated SAC policy.
        alpha (float): The entropy coefficient.
        data (dict): Dictionary containing observations.

    Returns:
        tuple: A tuple containing the policy loss and additional information.
     """
    o = data['obs']
    pi, logp_pi = updated_sac_policy.pi(o)
    q1_pi = updated_sac_policy.q1(o, pi)
    q2_pi = updated_sac_policy.q2(o, pi)
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
    loss_pi, _ = compute_loss_pi_for_explainableSac(sac_policy, alpha, observation_data)  # Compute policy loss

    return -loss_pi

def DGD_find_argmin_lagrangian(sac_policy, gbr_policy, alpha, lambda_val, observation_data, num_iterations=100, lr=0.001):
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
    sac_policy_copy = deepcopy(sac_policy)

    # Define optimizer
    optimizer = optim.Adam(sac_policy_copy.parameters(), lr=lr)

    for _ in range(num_iterations):
        # Calculate constraint term using gbr_policy
        constraint_term = DGD_calculate_constraint(sac_policy_copy, gbr_policy, observation_data)

        # Compute loss_pi using sac_policy_copy
        loss_pi, _ = compute_loss_pi_for_explainableSac(sac_policy, alpha, observation_data)

        # Compute Lagrangian
        lagrangian = -loss_pi + lambda_val * constraint_term

        # Zero gradients, perform backward pass, and update policy parameters
        optimizer.zero_grad()
        lagrangian.backward()
        optimizer.step()

    # Return policy parameters at the argmin of the Lagrangian
    return sac_policy_copy

def DGD_calculate_gradient(sac_policy, gbr_policy, observation_data):
     """
     Makes use of the updated sac_policy, which is the one that for a given lambda
     value, gives us the argmin Lagrangian through DGD_find_argmin_lagrangian
      """

    return DGD_calculate_constraint(sac_policy, gbr_policy, observation_data)

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

def prepare_data_for_gbr(sac_policy, replay_buffer, replay_buffer_size_to_sample, test_ratio=0.2):
    """
    Prepare data from a replay buffer for training a GradientBoostingRegressor (GBR) model.

    Args:
        replay_buffer (ReplayBuffer): The replay buffer containing the samples.
        test_ratio (float):           The ratio of samples to use for testing.

    Returns:
        tuple: A tuple containing X_train, X_test, y_train, and y_test.
    """
    # Sample from the replay buffer
    batch = replay_buffer.sample_batch(replay_buffer_size_to_sample * replay_buffer.size)

    # Extract observations and actions from the sampled batch
    obs = batch['obs']

    # Pass observations through the SAC policy to get actions We do that in order to have the 
    # result of the LAST policy. Otherwise when we sample from the replay buffer, we get (obs, action)
    # pairs that come from old policies
    actions = sac_policy(obs)

    # Flatten the observations
    obs_flat = obs.reshape((obs.shape[0], -1))

    # Split the data into training and testing sets
    num_samples = obs_flat.shape[0]
    num_test_samples = int(test_ratio * num_samples)

    X_train = obs_flat[num_test_samples:]
    X_test = obs_flat[:num_test_samples]
    y_train = actions[num_test_samples:]
    y_test = actions[:num_test_samples]

    return X_train, X_test, y_train, y_test

def train_gbr(X_train, y_train, **kwargs):
    """
    Train a Gradient Boosting Regressor (GBR) model using XGBoost.

    Args:
        X_train (array-like): The input features for training.
        y_train (array-like): The target labels for training.
        **kwargs: Additional keyword arguments to pass to the XGBoost model.

    Returns:
        Booster: The trained XGBoost model.
    """
    # Convert data to DMatrix format
    dtrain = xgb.DMatrix(X_train, label=y_train)

    # Train the XGBoost model
    gbr_model = xgb.train(params=kwargs, dtrain=dtrain)

    return gbr_model


