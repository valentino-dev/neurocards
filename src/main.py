import scipy
import numpy as np

HAND_SIZE = 7
DECK_COUNT = 1
PLAYER_COUNT = 2

color = np.arange(2, 15)
deck = np.repeat(color, 4 * DECK_COUNT)


def play_card(hand, played_cards, own_position, beginning_position, stich_idx):
    """
    Some algorythm to choose the card to play.
    Must return the index of the played card.
    """

    print(f"Current stich index: {stich_idx}")
    print(f"own position: {own_position},\t\tbeginning position: {beginning_position}")
    print(f"Hand: \n{np.array([np.arange(7), hand])}")
    print("Played Cards: \n", played_cards)

    played_card_idx = int(input("What card do you wanna play (index)?"))

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
    alive_players = np.zeros((PLAYER_COUNT)) + 1
    last_won_position = -1

    while not alive_players.sum() == 1:
        playing_cards = np.random.choice(
            deck, size=HAND_SIZE * PLAYER_COUNT, replace=False
        )
        hand = np.reshape(playing_cards, (PLAYER_COUNT, HAND_SIZE))

        played_cards = np.zeros((HAND_SIZE, PLAYER_COUNT))

        last_won_position = (last_won_position + 1) % PLAYER_COUNT

        for stich_idx in range(HAND_SIZE):
            for player_idx in range(PLAYER_COUNT):

                playing_player_idx = (player_idx + last_won_position) % PLAYER_COUNT

                if alive_players[playing_player_idx] == 1:
                    played_card_idx = play_card(
                        hand[playing_player_idx],
                        played_cards,
                        playing_player_idx,
                        last_won_position,
                        stich_idx,
                    )
                    played_cards[stich_idx, playing_player_idx] = hand[
                        playing_player_idx, played_card_idx
                    ]

                    hand[playing_player_idx, played_card_idx] = 0
            temp = np.array(played_cards[stich_idx])
            np.place(temp, temp == 0, 100)
            last_won_position = (
                (
                    PLAYER_COUNT
                    - 1
                    - np.argmax(
                        np.flip(
                            np.roll(
                                temp,
                                -last_won_position,
                            )
                        )
                    )
                )
                + last_won_position
            ) % PLAYER_COUNT
            print("\n")
            print(f"player {last_won_position} won the stich!")
            print("\n")
            print("\n")

        player_points[last_won_position] += played_cards[-1].sum()
        print("\n\n")
        print(player_points)
        print("\n\n")

        alive_players = player_points < 21


game(deck)
