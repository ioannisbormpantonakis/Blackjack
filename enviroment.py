import numpy as np

class Blackjack:
    ACTION_HIT = 0
    ACTION_STAY = 1
    NUM_DECKS = 4
    RESHUFFLE_CUTOFF = 13


    def __init__(self):
        self.player_hand = np.array([], dtype=int)
        self.dealer_hand = np.array([], dtype=int)
        self.deck = self.new_shuffled_deck()
        self.game_done = False


    def new_shuffled_deck(self):
        deck = np.array([i for i in range(1, 14) for _ in range(4 * self.NUM_DECKS)])
        np.random.shuffle(deck)
        return deck
    
    
    def draw_card(self):
        card = self.deck[0]
        self.deck = self.deck[1:]
        return card
    
    
    def hand_value(self, hand):
        total = np.sum([min(card, 10) for card in hand])
        num_aces = np.sum(hand == 1)
        while num_aces and (total + 10 <= 21):
            total += 10
            num_aces -= 1
        return total
    
    
    def usable_ace(self, hand):
        result = True if (1 in hand) and (np.sum([min(card, 10) for card in hand]) + 10 <= 21) else False
        return result
    

    def is_burned(self, hand):
        result = False if self.hand_value(hand) <= 21 else True
        return result 
    

    def get_state(self):
        state = {
            'player_total' : self.hand_value(self.player_hand),
            'usable_ace' : self.usable_ace(self.player_hand),
            'dealer_upcard' : self.dealer_hand[0]
        }
        return state


    def game_reset(self):
        if len(self.deck) <= self.RESHUFFLE_CUTOFF:
            self.deck = self.new_shuffled_deck()

        self.game_done = False
        self.reward = 0
        self.player_hand = np.array([self.draw_card(), self.draw_card()])
        self.dealer_hand = np.array([self.draw_card(), self.draw_card()])

        player_has_blackjack = self.hand_value(self.player_hand) == 21
        dealer_has_blackjack = self.hand_value(self.dealer_hand) == 21

        if player_has_blackjack and dealer_has_blackjack:
            self.reward = 0
            self.game_done = True
        if player_has_blackjack and not dealer_has_blackjack:                
            self.reward = 1.5
            self.game_done = True
        if not player_has_blackjack and dealer_has_blackjack:
            self.reward = -1
            self.game_done = True        

        return self.get_state()
    

    def player_step(self, action):

        if self.game_done:
            raise ValueError('Attempted to player step while game done')

        if action == self.ACTION_HIT:
            self.player_hand = np.append(self.player_hand, self.draw_card())
            if self.is_burned(self.player_hand):
                self.game_done = True
                self.reward = -1
                return (self.get_state(), self.reward, self.game_done)
            else:
                self.reward = 0
                return (self.get_state(), self.reward, self.game_done)
            
        if action == self.ACTION_STAY:
            while self.hand_value(self.dealer_hand) < 17:
                self.dealer_hand = np.append(self.dealer_hand, self.draw_card())
            if self.is_burned(self.dealer_hand):
                self.reward = 1
            else:
                player_total = self.hand_value(self.player_hand)
                dealer_total = self.hand_value(self.dealer_hand)
                self.reward = 1 if (player_total > dealer_total) else (-1 if (player_total < dealer_total) else 0)    
            self.game_done = True
            return (self.get_state(), self.reward, self.game_done)