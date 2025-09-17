import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import plotly.graph_objects as go


# Transition MAtrix Heatmap
def plot_transition_heatmap(matrix, title="Transition Matrix", figsize=(10, 8), cmap="Blues"):
    """
    Plots a heatmap for a given transition matrix.

    Parameters:
    - matrix (pd.DataFrame): A pandas DataFrame representing the transition matrix
    - title (str): Title of the heatmap
    - figsize (tuple): Size of the plot
    - cmap (str): Color map for the heatmap
    """
    plt.figure(figsize=figsize)
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap=cmap)
    plt.title(title, fontsize=14)
    plt.ylabel("From Channel")
    plt.xlabel("To Channel")
    plt.tight_layout()
    plt.show()


# R Matrix Heatmap
def plot_absorbing_heatmap(matrix, title="R Matrix: Transient → Absorbing Probabilities",figsize=(8, 6), cmap="Blues"):
    """
    Plots a heatmap for the R matrix showing transient to absorbing state probabilities.

    Parameters:
    - matrix (pd.DataFrame): Transition probability matrix (R matrix)
    - title (str): Title of the heatmap
    - figsize (tuple): Size of the plot
    - cmap (str): Color map for the heatmap
    """
    plt.figure(figsize=figsize)
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap=cmap)
    plt.title(title, fontsize=14)
    plt.ylabel("From Channel")
    plt.xlabel("To Absorbing State")
    plt.tight_layout()
    plt.show()

    
# Image display
def display_image(image_path, figsize=(10, 10), hide_axis=True):
    """
    Displays an image using matplotlib.

    Parameters:
    - image_path (str): Path to the image file
    - figsize (tuple): Size of the display figure
    - hide_axis (bool): Whether to hide axis ticks and labels
    """
    img = mpimg.imread(image_path)
    plt.figure(figsize=figsize)
    plt.imshow(img)
    if hide_axis:
        plt.axis('off')
    plt.show()


# Plot ranked channel
def plot_ranked_channels(ranked_channels, exclude_start=True, title='Ranked Channels by Expected Visits Before Absorption',palette='viridis'):
    """
    Plots a bar chart of ranked channels based on expected visits before absorption.

    Parameters:
    - ranked_channels (pd.Series): Series with channel names as index and expected visits as values
    - exclude_start (bool): Whether to drop 'Start' from the ranking
    - title (str): Title of the plot
    """
    if exclude_start:
        channel_imp = ranked_channels.drop('Start', errors='ignore')
    else:
        channel_imp = ranked_channels

    plt.figure(figsize=(10, 6))
    sns.barplot(x=channel_imp.index, y=channel_imp.values)
    plt.title(title)
    plt.xlabel('Channels')
    plt.ylabel('Expected Visits')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# Plot sankey diagram
def plot_sankey_from_matrix(df_counts):
    # Extract unique labels (channels)
    labels = list(df_counts.index)

    # Prepare source, target, and value lists for Sankey
    sources = []
    targets = []
    values = []

    label_to_index = {label: i for i, label in enumerate(labels)}

    for from_label in labels:
        for to_label in labels:
            count = df_counts.loc[from_label, to_label]
            if count > 0:
                sources.append(label_to_index[from_label])
                targets.append(label_to_index[to_label])
                values.append(count)

    # Create Sankey figure
    sankey_fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color="skyblue"
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color="lightgreen"
        ))])

    sankey_fig.update_layout(title_text="Sankey Diagram of Channel Transitions", font_size=12)
    sankey_fig.show()


# Plot conversion probability

def plot_conversion_probabilities(b_matrix, absorbing_state='conversion', figsize=(10, 6), palette='viridis'):
    """
    Plots a horizontal bar chart of channel conversion probabilities from the B matrix.

    Parameters:
    - b_matrix (pd.DataFrame): Absorption probability matrix (B = N × R)
    - absorbing_state (str): Column name representing the absorbing state (e.g., 'Conversion')
    - figsize (tuple): Size of the plot
    - palette (str): Color palette for the bars
    """
    conversion_probs = b_matrix[absorbing_state].sort_values(ascending=False)

    plt.figure(figsize=figsize)
    sns.barplot(x=conversion_probs.values, y=conversion_probs.index, palette=palette)
    plt.title(f"Channel Conversion Probabilities (from B matrix)", fontsize=14)
    plt.xlabel("Probability of Conversion")
    plt.ylabel("Channel")
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


# Plot channel removal
def plot_removal_effects(effects_series, title="Removal Effects on Conversion Probability (in pp)"):
    """
    Plots the removal effects of each channel on conversion probability.

    Args:
        effects_series (pd.Series): Series of removal effects (numeric values), indexed by channel.
        title (str): Title for the plot.
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(x=effects_series.values, y=effects_series.index, palette="viridis")
    plt.title(title, fontsize=14)
    plt.xlabel("Removal Effect (pp)")
    plt.ylabel("Channel")
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()