import numpy as np
import pandas as pd
from collections import Counter

# User Journey Function

def create_user_journeys(df):
    """
    Builds journey for each user

    Parameters:
    - df: Input DataFrame with columns 'channel', 'converted', 'session_time', 'user_id'

    Returns:
    - df_paths: Output DataFrame with user_id, channel_list, result, and user_journey
    """


    df['channel'] = df['channel'].str.lower()


    df['Timestamp'] = pd.to_datetime(df['session_time'])


    df_sorted = df.sort_values(by=['user_id', 'session_time'])


    df_paths = df_sorted.groupby("user_id").agg(list)


    df_paths.rename(columns={"channel": "channel_list"}, inplace=True)


    df_paths = df_paths.reset_index()


    df_paths["result"] = df_paths["converted"].apply(lambda lst: "conversion" if any(lst) else "dropped")

    # Build user journey add the start state as it will be needed for sankey diagram
    df_paths["user_journey"] = df_paths.apply(
        lambda p: ["start"] + [channel for channel in p["channel_list"]] + [p["result"]],
        axis=1
    )

    return df_paths


# Transition Matrix Function



def create_transition_matrix(df, df2, journey_col='user_journey', channel_col='channel'):
    """
    Creates the transition matrix, which shows both count and probabilities

    Parameters:
    - df : Original DataFrame containing channel information
    - df_paths : DataFrame containing user journeys
    - journey_col (str): Column name in df_paths containing user journeys
    - channel_col (str): Column name in df containing channel names

    Returns:
    - count_df (pd.DataFrame): Transition count matrix
    - prob_df (pd.DataFrame): Transition probability matrix
    """


    all_state = df[channel_col].unique()
    states = ['start'] + sorted(all_state.tolist()) + ['conversion', 'dropped']

    # indexing will make it easier to create the matrix with all interconnected states
    state_idx = {state: i for i, state in enumerate(states)}
    state_matrix_len = len(states)

    # creates the matrix that gives the count of one state to another
    count_matrix = np.zeros((state_matrix_len, state_matrix_len), dtype=int)

    for journey in df2[journey_col]:
        for i in range(len(journey) - 1):
            from_channel = journey[i]
            to_channel = journey[i + 1]
            count_matrix[state_idx[from_channel]][state_idx[to_channel]] += 1

    # Create DataFrame for count matrix
    count_df = pd.DataFrame(count_matrix, index=states, columns=states)

    # computes probability matrix
    prob_matrix = np.zeros_like(count_matrix, dtype=float)
    row_sum = count_matrix.sum(axis=1, keepdims=True)
    prob_matrix = np.divide(count_matrix, row_sum, out=prob_matrix, where=row_sum != 0)

    # Create DataFrame for probability matrix
    prob_df = pd.DataFrame(prob_matrix.round(4), index=states, columns=states)

    return count_df, prob_df


# Create Q Matrix

def extract_q_matrix(prob_df, absorbing_states):
  transient_states = [state for state in prob_df.index if state not in absorbing_states]
  q_df = prob_df.loc[transient_states, transient_states]
  return q_df


# Create R Matrix

def extract_r_matrix(prob_df, absorbing_states):

    transient_states = [state for state in prob_df.index if state not in absorbing_states]


    r_df = prob_df.loc[transient_states, absorbing_states]
    return r_df 


# Computes Fundamental Matrix

def compute_fundamental_matrix(q_matrix):

    transient_states = q_matrix.index.tolist()

    Q = q_matrix.values

    I = np.eye(len(Q))
    I_minus_Q = I - Q

    N = np.linalg.inv(I_minus_Q)

    N_df = pd.DataFrame(N, index=transient_states, columns=transient_states)

    return N_df


# Rank Channel 

def rank_channels_by_expected_visits(q_matrix, compute_fundamental_matrix_func):
    """
    Computes the fundamental matrix and ranks channels by expected visits before absorption.

    Parameters:
    - q_matrix (pd.DataFrame): Q matrix representing transient-to-transient transition probabilities
    - compute_fundamental_matrix_func (function): Function to compute the fundamental matrix

    Returns:
    - ranked_channels (pd.Series): Channels ranked by expected visits
    - n_df (pd.DataFrame): Fundamental matrix
    """
    # Compute fundamental matrix
    n_df = compute_fundamental_matrix_func(q_matrix)

    # Rank channels by expected visits (column-wise sum)
    ranked_channels = n_df.sum(axis=0).sort_values(ascending=False)

    print("Ranked Channels by Expected Visits Before Absorption:")
    print(ranked_channels)

    return ranked_channels, n_df


# Absorption probability function

def compute_absorption_probabilities(n_df, r_matrix):
    """
    Computes the absorption probability matrix B = N × R.

    Parameters:
    - n_df (pd.DataFrame): Fundamental matrix (N)
    - r_matrix (pd.DataFrame): R matrix representing transient → absorbing transitions

    Returns:
    - b (pd.DataFrame): Absorption probability matrix
    """
    b = n_df @ r_matrix
    b = b.round(4)

    print("Absorption probability matrix (B = N × R):")
    print(b)

    return b


# Conversion Probability Function

def get_conversion_probabilities(b_matrix, absorbing_state='conversion'):
    """
    Extracts and ranks channels by their probability of leading to a specific absorbing state.

    Parameters:
    - b_matrix (pd.DataFrame): Absorption probability matrix (B = N × R)
    - absorbing_state (str): Name of the absorbing state column to rank by


    Returns:
    - conversion_probs (pd.Series): Channels ranked by conversion probability
    """
    conversion_probs = b_matrix[absorbing_state].sort_values(ascending=False)


    print(f"Channel Conversion Probabilities (→ {absorbing_state}):")
    print(conversion_probs)

    return conversion_probs



# Removal Effect

def removal_effects(prob_df, absorbing_states, baseline_N, R_matrix):
    """
    Calculate removal effects of each transient channel on conversion probability.

    Args:
        prob_df (pd.DataFrame): Full transition probability matrix (states x states).
        absorbing_states (list): List of absorbing states (e.g., ['Conversion', 'Null']).
        baseline_N (pd.DataFrame): Fundamental matrix for full transient states.
        R_matrix (pd.DataFrame): Transition probabilities from transient to absorbing states.

    Returns:
        pd.Series: Removal effects in percentage points, indexed by channel, sorted descending.
    """
    transient_states = [s for s in prob_df.index if s not in absorbing_states and s != 'start']

    # Compute baseline conversion probability
    baseline_B = baseline_N @ R_matrix
    baseline_conversion = baseline_B['conversion'].sum()

    removal_effects = {}

    for channel in transient_states:
        reduced_states = [s for s in transient_states if s != channel]

        # Extract reduced Q and R matrices
        Q_reduced = prob_df.loc[reduced_states, reduced_states].values
        R_reduced = prob_df.loc[reduced_states, absorbing_states]

        # Compute reduced fundamental matrix
        I_reduced = np.eye(len(reduced_states))

        # in future handle the edgecase here
        N_reduced = np.linalg.inv(I_reduced - Q_reduced)


        # Compute reduced conversion probability
        B_reduced = N_reduced @ R_reduced
        reduced_conversion = B_reduced['conversion'].sum()

        # Removal effect in percentage points
        drop_pp = 100 * (baseline_conversion - reduced_conversion)
        removal_effects[channel] = round(drop_pp, 2)

    return pd.Series(removal_effects).sort_values(ascending=False).astype(str) + " pp"


# user_path simulation

def simulate_user_path(transition_matrix, start_state='start',absorbing_states=('conversion', 'dropped'), max_steps=10000):
    """
    Simulates a single user journey through the channel funnel using NumPy.
    """
    path = [start_state]
    current = start_state

    for _ in range(max_steps):
        if current in absorbing_states:
            break

        # Probabilities for next states from the current row
        probs = transition_matrix.loc[current].to_numpy(dtype=float)

        # Safety: normalize if not perfectly summing to 1; stop if all zeros
        s = probs.sum()
        if s <= 0:
            # No outgoing transitions → treat as terminal
            break
        probs = probs / s

        # Draw next state directly (returns a string, not a list)
        next_state = np.random.choice(transition_matrix.columns, p=probs)

        path.append(next_state)
        current = next_state

    return path


# Monte Carlo Simulation

def simulate_conversion_rate(transition_matrix, n_simulations=10000, start_state='start', absorbing_states=['conversion', 'dropped'], max_steps=10000):
    """
    Simulate multiple user journeys and return conversion rate.
    """
    conversions = 0
    for _ in range(n_simulations):
        path = simulate_user_path(transition_matrix, start_state=start_state, absorbing_states=absorbing_states, max_steps=max_steps)
        if path[-1] == 'conversion':
            conversions += 1
    return conversions / n_simulations


def simulate_multiple_paths(transition_matrix, n_simulations=10000, start_state='Start', absorbing_states=['conversion', 'dropped']):
    """
    Simulate multiple user journeys and count visits to channels (excluding absorbing states).
    """
    channel_counter = Counter()
    for _ in range(n_simulations):
        path = simulate_user_path(transition_matrix, start_state=start_state, absorbing_states=absorbing_states)
        for channel in path:
            if channel != start_state and channel not in absorbing_states:
                channel_counter[channel] += 1
    attribution_counts = pd.Series(channel_counter)
    attribution_percent = attribution_counts / attribution_counts.sum()
    return attribution_counts, attribution_percent



def compute_removal_effects_pp(transition_matrix, channels_to_remove=None, n_simulations=10000, start_state='start', absorbing_states=['conversion', 'dropped'], max_steps=10000):
    """
    Computes removal effects in percentage points for each channel.
    """
    if channels_to_remove is None:
        channels_to_remove = [ch for ch in transition_matrix.columns if ch not in absorbing_states and ch != start_state]

    baseline_rate = simulate_conversion_rate(transition_matrix, n_simulations, start_state, absorbing_states, max_steps)
    removal_effects = {}

    for channel in channels_to_remove:
        modified_matrix = transition_matrix.copy()

        # Remove channel by zeroing its row and column
        modified_matrix.loc[channel, :] = 0
        modified_matrix.loc[:, channel] = 0

        # Normalize rows to maintain valid probabilities
        modified_matrix = modified_matrix.div(modified_matrix.sum(axis=1), axis=0).fillna(0)

        new_rate = simulate_conversion_rate(modified_matrix, n_simulations, start_state, absorbing_states, max_steps)
        drop_pp = 100 * (baseline_rate - new_rate)
        removal_effects[channel] = round(drop_pp, 2)

    return pd.Series(removal_effects).sort_values(ascending=False).astype(str) + " pp"


