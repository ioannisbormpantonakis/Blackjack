import numpy as np

class RandomAgent: 
    def action(self, state):
        return np.random.choice([0, 1])
    
class HitUntilSeventeenAgent:
    def action(self, state):
        return 0 if state['player_total'] < 17 else 1    


#============================ Learning Agents ============================#

#============ Q-Learning Agent =============# 

class QLearningAgent:
    def __init__(self, params):
        self.alpha = params['alpha']
        self.epsilon = params['epsilon']
        self.gamma = params['gamma']
        self.current_step = 0

        self.q_table = np.zeros((18, 10, 2, 2))
        self.q_counter = np.zeros((18, 10, 2, 2))

    def get_state_indices(self, state):
        i_pt = state['player_total'] - 4
        i_du = min(state['dealer_upcard'] - 1, 9)
        i_ua = int(state['usable_ace'])
        return (i_pt, i_du, i_ua)
    
    def action(self, state):
        i_pt, i_du, i_ua = self.get_state_indices(state)
        epsilon = self.epsilon[self.current_step]

        if np.random.rand() < epsilon:
            return np.random.choice([0, 1])
        else:
            return np.argmax(self.q_table[i_pt, i_du, i_ua, :])
    
    def update(self, old_state, new_state, action, reward, game_done):
        i_pt_old, i_du_old, i_ua_old = self.get_state_indices(old_state)
        alpha = self.alpha[self.current_step]
        q_old = self.q_table[i_pt_old, i_du_old, i_ua_old, action]
        self.q_counter[i_pt_old, i_du_old, i_ua_old, action] += 1

        if game_done:
            target = reward
        else:
            i_pt_new, i_du_new, i_ua_new = self.get_state_indices(new_state)
            q_new = np.max(self.q_table[i_pt_new, i_du_new, i_ua_new, :])
            target = reward + self.gamma * q_new

        self.q_table[i_pt_old, i_du_old, i_ua_old, action] += alpha * (target - q_old)
        self.current_step += 1   
    

class QGreedyAgent:
    def __init__(self, q_agent):
        self.q_agent = q_agent
    
    def action(self, state):
        i_pt, i_du, i_ua = self.q_agent.get_state_indices(state)
        return np.argmax(self.q_agent.q_table[i_pt, i_du, i_ua, :])

#============ Sarsa Agent =============# 

class SarsaAgent:
    def __init__(self, params):
        self.alpha = params['alpha']
        self.epsilon = params['epsilon']
        self.gamma = params['gamma']
        self.current_step = 0

        self.q_table = np.zeros((18, 10, 2, 2))
        self.q_counter = np.zeros((18, 10, 2, 2))

    def get_state_indices(self, state):
        i_pt = state['player_total'] - 4
        i_du = min(state['dealer_upcard'] - 1, 9)
        i_ua = int(state['usable_ace'])
        return (i_pt, i_du, i_ua)
    
    def action(self, state):
        i_pt, i_du, i_ua = self.get_state_indices(state)
        epsilon = self.epsilon[self.current_step]

        if np.random.rand() < epsilon:
            return np.random.choice([0, 1])
        else:
            return np.argmax(self.q_table[i_pt, i_du, i_ua, :])
    
    def update(self, old_state, new_state, old_action, new_action, reward, game_done):
        i_pt_old, i_du_old, i_ua_old = self.get_state_indices(old_state)
        alpha = self.alpha[self.current_step]
        q_old = self.q_table[i_pt_old, i_du_old, i_ua_old, old_action]
        self.q_counter[i_pt_old, i_du_old, i_ua_old, old_action] += 1

        if game_done:
            target = reward
        else:
            i_pt_new, i_du_new, i_ua_new = self.get_state_indices(new_state)
            q_new = self.q_table[i_pt_new, i_du_new, i_ua_new, new_action]
            target = reward + self.gamma * q_new

        self.q_table[i_pt_old, i_du_old, i_ua_old, old_action] += alpha * (target - q_old)
        self.current_step += 1


class SarsaGreedyAgent:
    def __init__(self, sarsa_agent):
        self.sarsa_agent = sarsa_agent
    
    def action(self, state):
        i_pt, i_du, i_ua = self.sarsa_agent.get_state_indices(state)
        return np.argmax(self.sarsa_agent.q_table[i_pt, i_du, i_ua, :])