import numpy as np 
import matplotlib.pyplot as plt
from enviroment import Blackjack

def draw(winrate_list, reward_list):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    x = np.arange(len(winrate_list))
    y_winrate = np.cumsum(winrate_list)
    y_reward = np.cumsum(reward_list)

    ax[0].plot(x, y_winrate, label='Total Wins-Losses', color='darkgreen')
    ax[0].set_facecolor('lightgray')
    ax[0].set_title('Total Wins-Losses')
    ax[0].grid()

    ax[1].plot(x, y_reward, label='Cumulative Reward', color='darkgreen')
    ax[1].set_facecolor('lightgray')
    ax[1].set_title('Total Reward $')
    ax[1].grid()

    fig.set_facecolor('wheat')
    plt.show()

def play_blackjack():
    print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    print('$                                                                          $')
    print('$                                                                          $')
    print('$                                                                          $')
    print('$                         R L      C A S I N O                             $')
    print('$                                                                          $')
    print('$                                                                          $')
    print('$                                                                          $')
    print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    print('$$$$$$$$$$$$$$$$$$$$$$$    THE BLACKJACK GAME    $$$$$$$$$$$$$$$$$$$$$$$$$$$')
    print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')

    lexicon = {1:'A', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9', 10:'10', 11:'J', 12:'Q', 13:'K'}
    history_winrate = [0]
    history_rewards = [0]
    total_rounds_played = 0

    env = Blackjack()
    while True:
        total_rounds_played += 1
        state = env.game_reset()
        current_reward = 0

        player_hand_graphics = [lexicon[x] for x in env.player_hand]
        dealer_hand_graphics = [lexicon[x] for x in env.dealer_hand]

        print(f'\n --- Round: {total_rounds_played} ---')
        print(f'--- New Hand Dealt')
        print(f'Player Hand: {player_hand_graphics}    Value: {env.hand_value(env.player_hand)}')
        print(f'Dealer Upcard: {dealer_hand_graphics[0]}')
        print(f'Cards remaining in deck: {len(env.deck)}')
        print('-' * 30)

        if env.game_done:
            current_reward = env.reward
            if current_reward == 1.5:
                print('Blackjack! You win 1.5$')
                history_winrate.append(1)
            if current_reward == 0:
                print('Draw with dealer blackjack.')
                history_winrate.append(0)
            history_rewards.append(current_reward)

        else:
            while True:
                valid_actions = ['h', 's']
                action = input('Choose action - (h)it, (s)tay:')
                if action not in valid_actions:
                    print('Invalid choice. Try again.')
                    continue
                if action == 'h':
                    _, reward, game_done = env.player_step(Blackjack.ACTION_HIT)
                if action == 's':
                    _, reward, game_done = env.player_step(Blackjack.ACTION_STAY)    

                current_reward += reward
                print(f'\nAction ({action})')

                if game_done:
                    break 

                else:
                    player_hand_graphics = [lexicon[x] for x in env.player_hand]
                    dealer_hand_graphics = [lexicon[x] for x in env.dealer_hand]
                    print(f'Player Hand: {player_hand_graphics}    Value: {env.hand_value(env.player_hand)}')
                    print(f'Dealer Upcard: {dealer_hand_graphics[0]}')
                    print(f'Cards remaining in deck: {len(env.deck)}')
                    print('-' * 30)

            result = 1 if current_reward > 0 else (-1 if current_reward < 0 else 0)
            history_winrate.append(result)
            history_rewards.append(current_reward)
            player_hand_graphics = [lexicon[x] for x in env.player_hand]
            dealer_hand_graphics = [lexicon[x] for x in env.dealer_hand]
            print(f'|| Round Complete || Reward = {current_reward}$ ||')
            print(f'Player Hand: {player_hand_graphics}    Value: {env.hand_value(env.player_hand)}')
            print(f'Dealer Hand: {dealer_hand_graphics}    Value: {env.hand_value(env.dealer_hand)}')
            print(f'Cards remaining in deck: {len(env.deck)}')
            print(f'Total Rounds Played: {total_rounds_played}  ~  Wins-Losses: {np.sum(history_winrate)}  ~  Total Reward: {np.sum(history_rewards)}$')

        while True:
            again = input('\n Play another hand? (y/n): ')
            if again not in ['y', 'n']:
                print('Please select (y) for Yes or (n) for No')
                continue
            if again == 'y':
                break
            if again == 'n':
                while True:
                    show_plots = input('\n Would you like to see your results? (y/n): ')
                    if show_plots not in ['y', 'n']:
                        print('Please select either (y) for Yes or (n) for No')
                        continue
                    if show_plots == 'y':
                        print(f'Total Wins-Losses & Reward over {total_rounds_played} rounds: ')
                        draw(winrate_list=history_winrate, reward_list=history_rewards)
                    print('\n Hope you enjoyed your visit in RL Casino!')
                    return

play_blackjack()