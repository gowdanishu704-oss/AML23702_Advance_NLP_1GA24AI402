import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA

# Curated palette for clean visualization
TOPIC_COLORS = [
    "#4F46E5", "#06B6D4", "#10B981", "#F59E0B", "#EF4444",
    "#8B5CF6", "#EC4899", "#14B8A6", "#F97316", "#6366F1"
]

def plot_top_words_grid(topics_data: list[dict], max_words: int = 10):
    """
    Renders Plotly subplots of top keywords and probability values
    for each discovered latent topic.
    """
    n_topics = len(topics_data)
    cols = 2 if n_topics > 1 else 1
    rows = (n_topics + cols - 1) // cols

    subplot_titles = [t["label"] for t in topics_data]
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.12,
        horizontal_spacing=0.12
    )

    for i, topic in enumerate(topics_data):
        r = (i // cols) + 1
        c = (i % cols) + 1
        
        words = topic["words"][:max_words][::-1]
        weights = topic["weights"][:max_words][::-1]
        color = TOPIC_COLORS[i % len(TOPIC_COLORS)]

        fig.add_trace(
            go.Bar(
                x=weights,
                y=words,
                orientation='h',
                marker=dict(color=color, opacity=0.85),
                name=f"Topic {topic['topic_id']}",
                hovertemplate="Word: %{y}<br>Probability: %{x:.4f}<extra></extra>",
                showlegend=False
            ),
            row=r, col=c
        )
        fig.update_xaxes(title_text="P(word | topic)", row=r, col=c, tickformat=".3f")

    fig.update_layout(
        height=max(450, rows * 280),
        title_text="<b>Discovered Topic-Word Distributions (Probabilistic Weights)</b>",
        title_x=0.05,
        template="plotly_white",
        margin=dict(l=60, r=40, t=80, b=50),
        font=dict(family="Inter, Roboto, sans-serif")
    )
    return fig

def plot_topic_proportions(df_docs: pd.DataFrame, topics_data: list[dict]):
    """
    Plots a donut chart representing document volume distribution
    across the discovered topics.
    """
    topic_counts = df_docs["dominant_topic"].value_counts().sort_index()
    labels = []
    values = []
    colors = []
    
    for k in topic_counts.index:
        label = topics_data[k]["label"] if k < len(topics_data) else f"Topic {k}"
        labels.append(label)
        values.append(int(topic_counts[k]))
        colors.append(TOPIC_COLORS[k % len(TOPIC_COLORS)])

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.45,
        marker=dict(colors=colors),
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>Articles: %{value}<br>Proportion: %{percent}<extra></extra>"
    )])

    fig.update_layout(
        title_text="<b>Dominant Topic Breakdown Across Corpus</b>",
        title_x=0.05,
        template="plotly_white",
        height=400,
        margin=dict(l=30, r=30, t=60, b=30),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    return fig

def plot_doc_topic_heatmap(doc_topic_dist: np.ndarray, max_docs: int = 40):
    """
    Renders document-topic probability mixture matrix theta_{d, k}
    for a sample of documents.
    """
    sample_size = min(max_docs, doc_topic_dist.shape[0])
    matrix_subset = doc_topic_dist[:sample_size, :]
    n_topics = doc_topic_dist.shape[1]

    fig = go.Figure(data=go.Heatmap(
        z=matrix_subset,
        x=[f"Topic {k}" for k in range(n_topics)],
        y=[f"Doc #{i+1}" for i in range(sample_size)],
        colorscale="Viridis",
        colorbar=dict(title="P(z | d)"),
        hovertemplate="Document: %{y}<br>Topic: %{x}<br>Weight: %{z:.3f}<extra></extra>"
    ))

    fig.update_layout(
        title_text=f"<b>Document-Topic Mixture Heatmap (Sample of {sample_size} Docs)</b>",
        title_x=0.05,
        template="plotly_white",
        height=max(400, sample_size * 14),
        xaxis_title="Latent Topic",
        yaxis_title="Document Index",
        margin=dict(l=60, r=40, t=60, b=50)
    )
    return fig

def plot_intertopic_map(topic_word_dist: np.ndarray, doc_topic_dist: np.ndarray, topics_data: list[dict]):
    """
    Plots a 2D PCA projection of the topic centroids in vocabulary space.
    Bubble size corresponds to the aggregate prevalence of each topic.
    """
    n_topics = topic_word_dist.shape[0]
    if n_topics < 2:
        return None

    # Topic sizes based on document mass
    topic_mass = doc_topic_dist.sum(axis=0)
    topic_mass_normalized = (topic_mass / topic_mass.sum()) * 100.0

    # Project topics into 2D space via PCA
    pca = PCA(n_components=2, random_state=42)
    coords_2d = pca.fit_transform(topic_word_dist)

    var_explained = pca.explained_variance_ratio_ * 100.0

    plot_df = pd.DataFrame({
        "PC1": coords_2d[:, 0],
        "PC2": coords_2d[:, 1],
        "Topic": [t["label"] for t in topics_data],
        "Prevalence": np.round(topic_mass_normalized, 1),
        "Size": np.clip(topic_mass_normalized * 2.5, 20, 65)
    })

    fig = px.scatter(
        plot_df,
        x="PC1",
        y="PC2",
        text="Topic",
        size="Size",
        color="Topic",
        color_discrete_sequence=TOPIC_COLORS[:n_topics],
        hover_data={"Size": False, "Prevalence": True, "PC1": ":.3f", "PC2": ":.3f"}
    )
    fig.update_traces(textposition='top center')

    fig.update_layout(
        title_text="<b>Inter-Topic Semantic Distance Map (2D PCA Projection)</b>",
        title_x=0.05,
        template="plotly_white",
        height=450,
        xaxis_title=f"PC 1 ({var_explained[0]:.1f}% variance)",
        yaxis_title=f"PC 2 ({var_explained[1]:.1f}% variance)",
        showlegend=False,
        margin=dict(l=50, r=50, t=60, b=50)
    )
    return fig

def plot_live_inference(topic_mixture: np.ndarray, topics_data: list[dict]):
    """
    Displays the estimated posterior topic distribution P(z | document_new)
    for a live user input text.
    """
    labels = [t["label"] for t in topics_data]
    percentages = topic_mixture * 100.0

    df_inf = pd.DataFrame({
        "Topic": labels,
        "Probability": topic_mixture,
        "Percentage": percentages
    }).sort_values(by="Probability", ascending=True)

    fig = go.Figure(go.Bar(
        x=df_inf["Percentage"],
        y=df_inf["Topic"],
        orientation='h',
        marker=dict(
            color=df_inf["Percentage"],
            colorscale="Tealgrn",
            showscale=True,
            colorbar=dict(title="%")
        ),
        text=df_inf["Percentage"].apply(lambda p: f"{p:.1f}%"),
        textposition="inside",
        hovertemplate="<b>%{y}</b><br>Posterior Probability: %{x:.2f}%<extra></extra>"
    ))

    fig.update_layout(
        title_text="<b>Inferred Topic Mixture for Submitted Text</b>",
        title_x=0.05,
        template="plotly_white",
        height=320,
        xaxis_title="Posterior Probability (%)",
        margin=dict(l=60, r=40, t=60, b=40)
    )
    return fig
