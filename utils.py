import numpy as np
import matplotlib.pyplot as plt
from enviroment import Blackjack
from agents import RandomAgent, HitUntilSeventeenAgent



def run_simulation(num_rounds, agent):
    env = Blackjack()
    history_results = []
    history_rewards = []
    history_actions = []
    history_states = []

    for round in range(num_rounds):
        state = env.game_reset()
        current_reward = 0
        round_actions = []
        round_states = [state]

        if env.game_done:
            current_reward = env.reward

        else:
            while not env.game_done:
                action = agent.action(state)
                round_actions.append(action)
                state, reward, _ = env.player_step(action)
                round_states.append(state)
                current_reward += reward

        result = 1 if current_reward > 0 else (-1 if current_reward < 0 else 0)
        history_actions.append(round_actions)
        history_states.append(round_states)        
        history_results.append(result)
        history_rewards.append(current_reward)

    history_results = np.array(history_results)
    history_rewards = np.array(history_rewards)
    history = {
        'results' : history_results,
        'rewards' : history_rewards,
        'actions' : history_actions,
        'states' : history_states
    }    
    
    return history



def run_simulation_q(num_rounds, agent):
    env = Blackjack()
    history_results = []
    history_rewards = []
    history_actions = []
    history_states = []

    for round in range(num_rounds):
        state = env.game_reset()
        current_reward = 0
        round_actions = []
        round_states = [state]

        if env.game_done:
            current_reward = env.reward
        else:
            old_state = None
            old_action = None

            while not env.game_done:
                action = agent.action(state)
                round_actions.append(action)
                old_state, old_action = state, action

                new_state, reward, game_done = env.player_step(action)
                round_states.append(new_state)
                current_reward += reward

                if old_state is not None:
                    agent.update(old_state, new_state, old_action, reward, game_done)

                state = new_state    

        result = 1 if current_reward > 0 else (-1 if current_reward < 0 else 0)
        history_actions.append(round_actions)
        history_states.append(round_states)        
        history_results.append(result)
        history_rewards.append(current_reward)

    history_results = np.array(history_results)
    history_rewards = np.array(history_rewards)
    history = {
        'results' : history_results,
        'rewards' : history_rewards,
        'actions' : history_actions,
        'states' : history_states,
        'q_table' : agent.q_table

    }    
    
    return history



def run_simulation_sarsa(num_rounds, agent):
    env = Blackjack()
    history_results = []
    history_rewards = []
    history_actions = []
    history_states = []

    for round in range(num_rounds):
        state = env.game_reset()
        current_reward = 0
        round_actions = []
        round_states = [state]

        if env.game_done:
            current_reward = env.reward
        else:
            old_action = agent.action(state)

            while not env.game_done:
                old_state = state
                round_actions.append(old_action)

                new_state, reward, game_done = env.player_step(old_action)
                round_states.append(new_state)
                current_reward += reward

                new_action = agent.action(new_state) if not game_done else None
                agent.update(old_state, new_state, old_action, new_action, reward, game_done)

                state = new_state
                old_action = new_action

        result = 1 if current_reward > 0 else (-1 if current_reward < 0 else 0)
        history_actions.append(round_actions)
        history_states.append(round_states)        
        history_results.append(result)
        history_rewards.append(current_reward)

    history_results = np.array(history_results)
    history_rewards = np.array(history_rewards)
    history = {
        'results' : history_results,
        'rewards' : history_rewards,
        'actions' : history_actions,
        'states' : history_states,
        'q_table' : agent.q_table

    }    
    
    return history



def show_results(history):
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax = ax.ravel()
    x = np.arange(1, len(history['results']) + 1)
    x_usable_ace = np.arange(12, len(history['results']) + 1)

    y_winrate = np.cumsum(history['results'] == 1) / x
    y_drawrate = np.cumsum(history['results'] == 0) / x
    y_lossrate = np.cumsum(history['results'] == -1) / x
    y_wins_minus_losses = np.cumsum(history['results'])
    y_cum_reward = np.cumsum(history['rewards'])
    y_avg_reward_per_round = y_cum_reward / x

    ax[0].plot(x, y_cum_reward, label='Total Reward', color='darkgreen')
    ax[0].set_facecolor('lightgray')
    ax[0].set_title('Total Reward $')
    ax[0].grid()
    ax[0].legend()

    ax[1].plot(x, y_avg_reward_per_round, label='Average Reward per Round', color='darkgreen')
    ax[1].set_facecolor('lightgray')
    ax[1].set_title('Average Reward per Round $')
    ax[1].grid()
    ax[1].legend()

    ax[2].plot(x, y_wins_minus_losses, label='Total Wins minus Losses')
    ax[2].set_facecolor('lightgray')
    ax[2].set_title('Total Wins minus Losses')
    ax[2].grid()
    ax[2].legend()

    ax[3].plot(x, y_winrate, label='Winrate', color='green')
    ax[3].plot(x, y_drawrate, label='Drawrate', color = 'darkblue')
    ax[3].plot(x, y_lossrate, label='Lossrate', color = 'red')
    ax[3].set_facecolor('lightgray')
    ax[3].set_title('Win/Draw/Loss-rates')
    ax[3].grid()
    ax[3].legend()

    fig.set_facecolor('wheat')
    plt.tight_layout()
    plt.show()



def show_prints(history):
    print('=======================  RESULTS  =======================')
    print(f'Total Rounds: {len(history["rewards"])}')
    print(f'Final Reward: {np.sum(history["rewards"]):.2f}$')
    print(f'Average Reward per Round: {np.mean(history["rewards"]):.4f}$')
    print(f'Winrate: {np.sum(history["results"] == 1) / len(history["results"]):.4f}  ||  Drawrate: {np.sum(history["results"] == 0) / len(history["results"]):.4f}  ||  Lossrate: {np.sum(history["results"] == -1) / len(history["results"]):.4f}')    



def show_agent_policy(agent):
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    player_totals_ua_false = np.arange(4, 22)
    player_totals_ua_true = np.arange(12, 22)
    dealer_upcards = np.arange(1, 11)

    for k, usable_ace in enumerate([False, True]):
        player_totals = player_totals_ua_true if usable_ace else player_totals_ua_false
        policy = np.zeros((len(player_totals), len(dealer_upcards)))
        for i, player_total in enumerate(player_totals):
            for j, dealer_upcard in enumerate(dealer_upcards):
                state = {
                    'player_total' : player_total,
                    'dealer_upcard' : dealer_upcard,
                    'usable_ace' : usable_ace
                }
                policy[i, j] = agent.action(state)

        xmin, xmax = (0.5, 10.5)
        ymin, ymax = (11.5, 21.5) if usable_ace else (3.5, 21.5)
        im = ax[k].imshow(policy, cmap='viridis', extent=[xmin, xmax, ymin, ymax], origin='lower')
        ax[k].set_facecolor('lightgray')
        ax[k].set_title(f'Usable Ace: {usable_ace}')
        ax[k].set_xlabel(f'Dealer Upcard')
        ax[k].set_ylabel(f'Player Total')
        ax[k].set_xticks(dealer_upcards)
        ax[k].set_yticks(player_totals)
        ax[k].set_xticks(dealer_upcards + 0.5, minor=True)
        ax[k].set_yticks(player_totals + 0.5, minor=True)
        ax[k].grid(which='minor')
    fig.set_facecolor('lightgray')
    fig.suptitle(f'Policy for Agent: {agent.__class__.__name__}')
    plt.show()



def show_q_agent_counter(q_agent):
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    player_totals_ua_false = np.arange(4, 22)
    player_totals_ua_true = np.arange(12, 22)
    dealer_upcards = np.arange(1, 11)

    for k, usable_ace in enumerate([False, True]):
        player_totals = player_totals_ua_true if usable_ace else player_totals_ua_false
        counter = np.zeros((len(player_totals), len(dealer_upcards)))
        for i, player_total in enumerate(player_totals):
            for j, dealer_upcard in enumerate(dealer_upcards):
                counter[i, j] = np.sum(q_agent.q_counter[i, j, k, :])

        xmin, xmax = (0.5, 10.5)
        ymin, ymax = (11.5, 21.5) if usable_ace else (3.5, 21.5)
        im = ax[k].imshow(counter, cmap='inferno', extent=[xmin, xmax, ymin, ymax], origin='lower')
        ax[k].set_facecolor('lightgray')
        ax[k].set_title(f'Usable Ace: {usable_ace}')
        ax[k].set_xlabel(f'Dealer Upcard')
        ax[k].set_ylabel(f'Player Total')
        ax[k].set_xticks(dealer_upcards)
        ax[k].set_yticks(player_totals)
        ax[k].set_xticks(dealer_upcards + 0.5, minor=True)
        ax[k].set_yticks(player_totals + 0.5, minor=True)
        ax[k].grid(which='minor')
    fig.colorbar(im, ax=ax[1])   
    fig.set_facecolor('lightgray')
    fig.suptitle(f'State visit counts for Agent: {q_agent.__class__.__name__}')
    plt.show()