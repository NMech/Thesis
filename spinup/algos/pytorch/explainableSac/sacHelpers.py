import torch

# Set up function for computing SAC Q-losses
def compute_loss_q(ac, ac_targ, alpha, gamma, data):
    o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

    q1 = ac.q1(o,a)
    q2 = ac.q2(o,a)

    # Bellman backup for Q functions
    with torch.no_grad():
        # Target actions come from *current* policy
        a2, logp_a2 = ac.pi(o2)

        # Target Q-values
        q1_pi_targ = ac_targ.q1(o2, a2)
        q2_pi_targ = ac_targ.q2(o2, a2)
        q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
        backup = r + gamma * (1 - d) * (q_pi_targ - alpha * logp_a2)

    # MSE loss against Bellman backup
    loss_q1 = ((q1 - backup)**2).mean()
    loss_q2 = ((q2 - backup)**2).mean()
    loss_q = loss_q1 + loss_q2

    # Useful info for logging
    q_info = dict(Q1Vals=q1.detach().numpy(),
                    Q2Vals=q2.detach().numpy())

    return loss_q, q_info


# Set up function for computing SAC pi loss
def compute_loss_pi(ac, alpha, data):
    o = data['obs']
    pi, logp_pi = ac.pi(o)
    q1_pi = ac.q1(o, pi)
    q2_pi = ac.q2(o, pi)
    q_pi = torch.min(q1_pi, q2_pi)

    # Entropy-regularized policy loss
    loss_pi = (alpha * logp_pi - q_pi).mean()

    # Useful info for logging
    pi_info = dict(LogPi=logp_pi.detach().numpy())

    return loss_pi, pi_info

def update(ac, ac_targ, alpha, gamma, pi_optimizer, q_optimizer, q_params, logger, polyak, data):
    # First run one gradient descent step for Q1 and Q2
    q_optimizer.zero_grad()
    loss_q, q_info = compute_loss_q(ac, ac_targ, alpha, gamma, data)
    loss_q.backward()
    q_optimizer.step()

    # Record things
    logger.store(LossQ=loss_q.item(), **q_info)

    # Freeze Q-networks so you don't waste computational effort 
    # computing gradients for them during the policy learning step.
    for p in q_params:
        p.requires_grad = False

    # Next run one gradient descent step for pi.
    pi_optimizer.zero_grad()
    loss_pi, pi_info = compute_loss_pi(ac, alpha, data)
    loss_pi.backward()
    pi_optimizer.step()

    # Unfreeze Q-networks so you can optimize it at next DDPG step.
    for p in q_params:
        p.requires_grad = True

    # Record things
    logger.store(LossPi=loss_pi.item(), **pi_info)

    # Finally, update target networks by polyak averaging.
    with torch.no_grad():
        for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
            # NB: We use an in-place operations "mul_", "add_" to update target
            # params, as opposed to "mul" and "add", which would make new tensors.
            p_targ.data.mul_(polyak)
            p_targ.data.add_((1 - polyak) * p.data)

def get_action(o, ac, deterministic=False):
    #return ac.act(torch.as_tensor(o, dtype=torch.float32), 
    #             deterministic)
    action = ac.act(torch.as_tensor(o, dtype=torch.float32), deterministic)
    print("Actionnnnnnnnnnnnn:", action)  # Print the action before returning    #NIKOS
    return action

def test_agent(ac, test_env, num_test_episodes, max_ep_len, logger):
    for j in range(num_test_episodes):
        o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
        while not(d or (ep_len == max_ep_len)):
            # Take deterministic actions at test time 
            o, r, d, _ = test_env.step(get_action(o, ac, True))
            ep_ret += r
            ep_len += 1
        logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)