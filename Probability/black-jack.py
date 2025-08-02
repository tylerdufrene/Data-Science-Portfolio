import pandas as pd
import numpy as np
import pydealer as pyd
import altair as alt
import streamlit as st

values = {
    "Ace": [1,11],
    "King": 10,
    "Queen": 10,
    "Jack": 10,
    "10": 10,
    "9": 9,
    "8": 8,
    "7": 7,
    "6": 6,
    "5": 5,
    "4": 4,
    "3": 3,
    "2": 2
}

counts = {
    'Ace': 4,
    "King": 4,
    "Queen": 4,
    "Jack": 4,
    "10": 4,
    "9": 4,
    "8": 4,
    "7": 4,
    "6": 4,
    "5": 4,
    "4": 4,
    "3": 4,
    "2": 4
}

suit_symbols = {
    "Spades": "♠",
    "Hearts": "♥",
    "Diamonds": "♦",
    "Clubs": "♣"
}
# card_label = f"{card.value}{suit_symbols[card.suit]}"
# st.write(card_label)

# 1. game setup player and dealer get a card. Player gets a second card

def deal_cards():
    ''' 
    Builds the deck, shuffles the deck, then distributes two card to the player
    and one card to the dealer. 

    Returns the dealer hand, player hand, and the deck
    '''
    deck = pyd.Deck() 
    deck.shuffle()
    player = pyd.Stack()
    dealer = pyd.Stack()
    player.add(deck.deal(1))
    dealer.add(deck.deal(1))
    player.add(deck.deal())
    return deck, player, dealer

def manually_select_cards(card_list):
    deck = pyd.Deck()
    deck.shuffle()
    player = pyd.Stack()
    dealer = pyd.Stack()
    player.add(deck.get_list(card_list))
    dealer.add(deck.deal(1))
    return deck, player, dealer

def count_card(card, hand_total=0):
    '''
    Counts and returns the value of the card given.

    If the total value of the hand is > 11, then receiving an Ace will count that as 1
    otherwise the ace will count as 11
    '''
    if (card.value == 'Ace') and (hand_total + 11 > 21):
        return values[card.value][0] 
    elif (card.value == 'Ace') and (hand_total + 11 <= 21):
        return values[card.value][1]
    else:
        return values[card.value]

def get_hand_value(cards):
    ''' 
    Returns the total value of a given hand
    '''
    hand_total = 0
    if len(cards) == 1:
        return count_card(cards[0])
    elif len(cards) < 1:
        assert 'No cards passed to function'
    else:
        for card in cards:
            hand_total += count_card(card, hand_total)
    return hand_total

st.title('Black Jack | Probability Simulator')

option = st.selectbox(label='Select Cards Manually or Draw Randomly', options=['Random','Manual Select'])

if 'deck' not in st.session_state:
    st.session_state.deck, st.session_state.player, st.session_state.dealer = deal_cards()


if option == 'Random':
    button = st.button('Generate Random Cards')
    if button: 
        st.session_state.deck, st.session_state.player, st.session_state.dealer = deal_cards()
        # player_cards = [(card.value, suit_symbols[card.suit]) for card in player.cards]
        # card_str = f'{player_cards[0][0]}{player_cards[0][1]} & {player_cards[1][0]}{player_cards[1][1]} = {get_hand_value(player.cards)}'
        # st.metric(label='Your Cards: ', value=card_str)
else:
    st.write('Select Cards (eg. 7 of Hearts, Ace of Spades, etc.)')
    col1, col2 = st.columns(2)
    with col1:
        st.write("Card 1")
        card1_suit = st.selectbox(label='Choose Suit 1', options=suit_symbols.keys())
        card1_value = st.selectbox(label='Choose Card Value 1', options=values.keys())
        card1 = f'{card1_value} of {card1_suit}'
    with col2:
        st.write("Card 2")
        card2_suit = st.selectbox(label='Choose Suit 2', options=suit_symbols.keys())
        card2_value = st.selectbox(label='Choose Card Value 2', options=values.keys())
        card2 = f'{card2_value} of {card2_suit}'

    card_list = [card1, card2]
    # deck, player, dealer = manually_select_cards(card_list)
    if card1 == card2:
        st.write(':red[Please do not choose the same card twice]')
    elif (card1 and card2) and (card1 != card2):
        st.session_state.deck, st.session_state.player, st.session_state.dealer = manually_select_cards([card1, card2])
        # player_cards = [(card.value, suit_symbols[card.suit]) for card in player.cards]
        # card_str = f'{player_cards[0][0]}{player_cards[0][1]} & {player_cards[1][0]}{player_cards[1][1]}'
        # st.write('Cards Found!')
        # st.metric(label='Your Cards: ', value=card_str)


def simulate_dealer(dealer_cards, player_cards, deck, write_cards=False):
    dealer_total = get_hand_value(dealer_cards)
    player_total = get_hand_value(player_cards)
    dealer_hit_cards = pyd.Stack()
    while dealer_total < 17:
        dealer_hit_cards.add(deck.deal(1))
        if (dealer_hit_cards.cards[-1].value == 'Ace') and (dealer_total + 11 > 21):
            card_value = 1
        else:
            card_value = count_card(dealer_hit_cards[-1])
        dealer_total += card_value
    # print(dealer_hit_cards.cards)
    # print(dealer_total)
    if write_cards:
        if len(dealer_hit_cards.cards) > 1:
            dealer_str = ' & '.join([f'{c.value}{suit_symbols[c.suit]}' for c in dealer_hit_cards.cards])
        else:
            dealer_str = f'{dealer_hit_cards.cards[0].value}{suit_symbols[dealer_hit_cards.cards[0].suit]}'
        st.write(f'Dealer Hit Cards: {dealer_str} = {get_hand_value(dealer_hit_cards.cards)}')
    deck.add(dealer_hit_cards.empty(return_cards=True))
    deck.shuffle()
    return dealer_total, player_total, deck

def simulate_player_hit(dealer_cards, player_cards, deck):
    ''' 
    Simulates the outcome of the game if the player stands
    '''
    dealer_total = get_hand_value(dealer_cards)
    player_total = get_hand_value(player_cards)
    # print('player_total', player_total)
    dealer_hit_cards = pyd.Stack()
    hit_cards = pyd.Stack()
    hit_cards.add(deck.deal(1))
    # print(hit_cards)
    card_value = count_card(hit_cards[-1], player_total)
    player_total += card_value
    # print(player_total)
    # print('dealer_total', dealer_total)
    while dealer_total < 17:
        dealer_hit_cards.add(deck.deal(1))
        card_value_dealer = count_card(dealer_hit_cards[-1])
        dealer_total += card_value_dealer
    # print(dealer_hit_cards.cards)
    # print(dealer_total)
    deck.add(dealer_hit_cards.empty(return_cards=True))
    deck.add(hit_cards.empty(return_cards=True))
    deck.shuffle()
    return dealer_total, player_total, deck

def outcomes(dealer_total, player_total):
    if player_total > 21: 
        return -1, 'Player Busts'
    elif dealer_total > 21: 
        return 1, 'Dealer Busts'
    elif player_total > dealer_total:
        return 1, 'Player Wins'
    elif player_total < dealer_total:
        return -1, 'Dealer Wins'
    else:
        return 0, 'Push'


def simulate_stand(n, dealer_hand, player_hand, deck):
    results = []
    for _ in range(n):
        dealer, player, _ = simulate_dealer(dealer_hand, player_hand, deck)
        result_int, result_str = outcomes(dealer, player)
        results.append((result_int, result_str))
    return results

def simulate_hit(n, dealer_hand, player_hand, deck):
    results = []
    for _ in range(n):
        dealer_hit, player_hit, deck = simulate_player_hit(dealer_hand, player_hand, deck)
        outcome = outcomes(dealer_hit, player_hit)
        # print(dealer_hit, player_hit, outcome)
        results.append(outcome)
    return results



try:
    player = st.session_state.player
    dealer = st.session_state.dealer
    deck = st.session_state.deck

    player_str = ' & '.join([f'{c.value}{suit_symbols[c.suit]}' for c in player.cards])
    dealer_str = f'{dealer.cards[0].value}{suit_symbols[dealer.cards[0].suit]}'

    st.metric('Your Hand', f'{player_str} = {get_hand_value(player.cards)}')
    st.metric('Dealer Shows', dealer_str)


    st.divider()

    stand_outcomes = simulate_stand(1000, dealer.cards, player.cards, deck)
    hit_outcomes = simulate_hit(1000, dealer.cards, player.cards, deck)

    df = pd.DataFrame({
            'x':np.arange(0,1000),
            'y_stand':[stand_outcomes[i][0] for i in range(len(stand_outcomes))],
            'y_stand_descriptive':[stand_outcomes[i][1] for i in range(len(stand_outcomes))],
            'y_hit': [hit_outcomes[i][0] for i in range(len(hit_outcomes))],
            'y_hit_descriptive':[hit_outcomes[i][1] for i in range(len(hit_outcomes))],
        })
    
    # st.dataframe(df.head())
    df['stand_cumsum'] = df['y_stand'].cumsum()
    df['hit_cumsum'] = df['y_hit'].cumsum()

    def get_ev(df, option):
        n = len(df)
        win = len(df[df[f'y_{option}']==1]) / n
        push = len(df[df[f'y_{option}']==0]) / n
        loss = len(df[df[f'y_{option}']==-1]) / n
        return win, push, loss, win + 0 + -1 * loss

    stand_win, stand_push, stand_loss, stand_ev = get_ev(df, 'stand')
    hit_win, hit_push, hit_loss, hit_ev = get_ev(df, 'hit')

    # st.write(f'{stand_win}')
    ev_df = pd.DataFrame({
        'Decision':['Stand','Hit'],
        'Prob(Win) = 1':[f'{stand_win:.1%}', f'{hit_win:.1%}'],
        'Prob(Push) = 0':[f'{stand_push:.1%}', f'{hit_push:.1%}'],
        'Prob(Loss) = -1':[f'{stand_loss:.1%}', f'{hit_loss:.1%}'],
        'Expected Value':[f'{stand_ev:.2f}', f'{hit_ev:.2f}']
    })
    st.dataframe(ev_df, use_container_width=True)

    y_min = min([df.stand_cumsum.min(), df.hit_cumsum.min()])
    y_max = max([df.stand_cumsum.max(), df.hit_cumsum.max()])

    y_scale = alt.Scale(domain=[y_min, y_max])

    stand = alt.Chart(df).mark_area().encode(
        x=alt.X('x', title='Number of Simulations'),
        y=alt.Y('stand_cumsum', scale=y_scale, title='Outcome of Stand (1 Win | 0 Push | -1 Loss)')
    ).properties(
        title='Cumulative Winnings of Stand'
    )
    hit = alt.Chart(df).mark_area().encode(
        x=alt.X('x', title='Number of Simulations'),
        y=alt.Y('hit_cumsum', scale=y_scale, title='Outcome of Hit (1 Win | 0 Push | -1 Loss)')
    ).properties(
        title='Cumulative Winnings of Hit'
    )

    final_chart = (stand | hit).resolve_scale(
        y='shared'
    )
    st.altair_chart(final_chart, use_container_width=True)

    st.divider()

    st.title('Play Out Hand!')

    move = st.selectbox('Select Your Next Move', options=['Stand','Hit'])
    if st.button("Play Out!", key=''):
        if move == 'Stand':
            dealer_total, player_total, deck = simulate_dealer(dealer.cards, player.cards, deck, write_cards=True)
            outcome_val, outcome_str = outcomes(dealer_total, player_total)
            st.write(f'Final Dealer Total: {dealer_total}')
            st.write(f'Final Player Total: {player_total}')
        else:
            dealer_total = get_hand_value(dealer.cards)
            player_total = get_hand_value(player.cards)
            player_hit_cards = pyd.Stack()
            player_hit_cards.add(deck.deal(1))
            
            if (player_hit_cards.cards[0].value == 'Ace') and (player_total + 11 > 21):
                player_total += 1
                hit_str = f'Ace {suit_symbols[player_hit_cards.cards[0].suit]}'
                st.metric('Your Draw', f'{hit_str} = 1')
            else:
                player_total += get_hand_value(player_hit_cards)
                hit_str = f'{player_hit_cards.cards[0].value}{suit_symbols[player_hit_cards.cards[0].suit]}'
                st.metric('Your Draw', f'{hit_str} = {get_hand_value(player_hit_cards.cards)}')
            
            dealer_total, _, deck = simulate_dealer(dealer.cards, player.cards, deck, write_cards=True)
            st.write(f'Final Dealer Total: {dealer_total}')
            st.write(f'Final Player Total: {player_total}')
            outcome_val, outcome_str = outcomes(dealer_total, player_total)
        st.success(outcome_str)



except Exception as e:
    st.write(e)

if st.button("🔄 Reset Game"):
    for key in ['deck', 'player', 'dealer']:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()










