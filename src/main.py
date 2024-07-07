import scipy
import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


HAND_SIZE = 7
DECK_COUNT = 1
PLAYER_COUNT = 3

color = np.arange(2, 15)
deck = np.repeat(color, 4 * DECK_COUNT)

# Define the neural network model
def build_model(state_shape, action_space):
    model = Sequential()
    model.add(Dense(24, input_shape=state_shape, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(action_space, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model

# Define the DQN agent
class DQNAgent:
    def __init__(self, state_shape, action_space):
        self.state_shape = state_shape
        self.action_space = action_space
        self.model = build_model(state_shape, action_space)
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        print("test", self.epsilon)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        print("testtttt", state.shape)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                print(next_state.shape)
                target = reward + self.gamma * np.amax(self.model.predict(nnext_state)[0])
            print("test", state.shape)
            target_f = self.model.predict(state)
            print(int(action))
            print(target_f.shape)
            target_f[0][int(action)] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

action_space = HAND_SIZE
state_shape = (1, HAND_SIZE + PLAYER_COUNT*HAND_SIZE + 3)
agent = DQNAgent(state_shape, action_space)
batch_size = 32
reward = np.zeros(PLAYER_COUNT)
last_action = np.zeros(PLAYER_COUNT)
state = np.zeros((PLAYER_COUNT*2, 1, state_shape[1]))
print(agent.model.summary())

def play_card(hand, played_cards, own_position, beginning_position, stich_idx):
    """
    Some algorythm to choose the card to play.
    Must return the index of the played card.
    """
    
    '''
    print(f"Current stich index: {stich_idx}")
    print(f"own position: {own_position},\t\tbeginning position: {beginning_position}")
    print(f"Hand: \n{np.array([np.arange(7), hand])}")
    print("Played Cards: \n", played_cards)

    played_card_idx = int(input("What card do you wanna play (index)?"))
    '''

    state = np.expand_dims(np.concatenate((hand, played_cards, own_position, beginning_position, stich_idx), axis=None), axis=0)
    played_card_idx = agent.act(state)


    temp = np.array(hand)
    np.place(temp, temp == 0, 100)
    if (
        (not beginning_position == own_position)
        and hand[played_card_idx] < played_cards[stich_idx].max()
    ) or hand[played_card_idx] == 0:
        played_card_idx = np.argmin(temp)
    return played_card_idx



def game(deck):
    player_points = np.zeros((PLAYER_COUNT))
    alive_players = player_points < 21
    last_won_position = -1

    while not alive_players.sum() == 1:
        playing_cards = np.random.choice(deck, size=HAND_SIZE * PLAYER_COUNT, replace=False)
        hand = np.reshape(playing_cards, (PLAYER_COUNT, HAND_SIZE))
        played_cards = np.zeros((HAND_SIZE, PLAYER_COUNT))
        last_won_position = (last_won_position + 1) % PLAYER_COUNT

        for stich_idx in range(HAND_SIZE):
            for player_idx in range(PLAYER_COUNT):

                playing_player_idx = (player_idx + last_won_position) % PLAYER_COUNT
                if alive_players[playing_player_idx] == 1:
                    current_hand = hand[playing_player_idx]

                    new_state = np.expand_dims(np.concatenate((current_hand, played_cards, player_idx, last_won_position, stich_idx), axis=None), axis=0)
                    #print(new_state.shape)

                    if stich_idx != 0:
                        #print(state[playing_player_idx].shape)

                        agent.remember(state[playing_player_idx], last_action[playing_player_idx], reward[playing_player_idx], new_state, False)
                        if len(agent.memory) > batch_size:
                            agent.replay(batch_size)
                    print("testtt")
                    action = agent.act(new_state)
                    temp = np.array(current_hand)
                    np.place(temp, temp == 0, 100)
                    if ((not last_won_position == playing_player_idx) and current_hand[action] < played_cards[stich_idx].max()) or current_hand[action] == 0:
                        played_card_idx = np.argmin(temp)
                    else:
                        played_card_idx = action

                    last_action[playing_player_idx] = action
                    played_cards[stich_idx, playing_player_idx] = current_hand[played_card_idx]
                    current_hand[played_card_idx] = 0

                    reward[playing_player_idx] = 0
                    if stich_idx != HAND_SIZE-1:
                        print(new_state.shape)
                        print(state.shape)
                        state[playing_player_idx] = new_state
                    else:
                        state[playing_player_idx + PLAYER_COUNT] = state[playing_player_idx]
                        state[playing_player_idx] = new_state
            temp = np.array(played_cards[stich_idx])
            np.place(temp, temp == 0, 100)
            last_won_position = (
                (PLAYER_COUNT - 1 - np.argmax(np.flip(np.roll(temp,-last_won_position)))) + last_won_position) % PLAYER_COUNT
            '''
            print("\n")
            print(f"player {last_won_position} won the stich!")
            print("\n")
            print("\n")
            '''

        player_points[last_won_position] += played_cards[-1].sum()
        '''
        print("\n\n")
        print(player_points)
        print("\n\n")
        '''

        new_alive_players = player_points < 21
        for i in range(PLAYER_COUNT):
            print(new_alive_players, alive_players)
            if not new_alive_players[i] and alive_players[i]:
                reward[i] += -10
                done = True
            else:
                done = False
            agent.remember(state[i + PLAYER_COUNT], last_action[i], reward[i], state[i], done)
        alive_players = new_alive_players

    player_won_idx = np.argmax(alive_players)
    reward[player_won_idx] += 100
    agent.remember(state[player_won_idx + PLAYER_COUNT], last_action[player_won_idx], reward[player_won_idx], state[player_won_idx], True)

    

game(deck)
