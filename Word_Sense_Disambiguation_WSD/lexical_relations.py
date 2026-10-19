"""
Lexical Relations Explorer & WordNet Visualizer
Extracts taxonomic and lexical relations from WordNet:
Synsets, Hypernyms, Hyponyms, Meronyms, Holonyms, Antonyms, Coordinate Terms, and Derivational Forms.
"""

from typing import List, Dict, Any, Optional
import nltk
from nltk.corpus import wordnet as wn
import pandas as pd
import networkx as nx
import plotly.graph_objects as go


def get_all_synsets(word: str, pos: Optional[str] = None) -> List[Any]:
    """Retrieve all WordNet synsets for a target word, optionally filtered by POS."""
    pos_map = {
        "noun": wn.NOUN,
        "verb": wn.VERB,
        "adj": wn.ADJ,
        "adv": wn.ADV,
        "n": wn.NOUN,
        "v": wn.VERB,
        "a": wn.ADJ,
        "r": wn.ADV
    }
    wn_pos = pos_map.get(pos.lower()) if pos else None
    if wn_pos:
        return wn.synsets(word.lower(), pos=wn_pos)
    return wn.synsets(word.lower())


def build_sense_inventory(word: str) -> pd.DataFrame:
    """
    Build a comprehensive sense inventory table for an ambiguous word.
    Includes Synset ID, POS, Definition, Synonyms/Lemmas, Example Sentences,
    and Lexical Relation counts.
    """
    synsets = get_all_synsets(word)
    rows = []
    for idx, syn in enumerate(synsets, start=1):
        lemmas = [lemma.name().replace("_", " ") for lemma in syn.lemmas()]
        antonyms = []
        for lemma in syn.lemmas():
            for ant in lemma.antonyms():
                antonyms.append(ant.name().replace("_", " "))
        
        hypernyms = [h.name().split(".")[0].replace("_", " ") for h in syn.hypernyms()]
        hyponyms = [h.name().split(".")[0].replace("_", " ") for h in syn.hyponyms()]
        part_meronyms = [m.name().split(".")[0].replace("_", " ") for m in syn.part_meronyms()]

        rows.append({
            "Sense #": idx,
            "Synset ID": syn.name(),
            "POS": syn.pos(),
            "Lemmas / Synonyms": ", ".join(lemmas),
            "Definition (Gloss)": syn.definition(),
            "Examples": " | ".join(syn.examples()) if syn.examples() else "None",
            "Direct Hypernyms": ", ".join(hypernyms[:5]) if hypernyms else "None",
            "Hyponym Count": len(syn.hyponyms()),
            "Meronyms": ", ".join(part_meronyms[:4]) if part_meronyms else "None",
            "Antonyms": ", ".join(set(antonyms)) if antonyms else "None"
        })
    return pd.DataFrame(rows)


def get_detailed_lexical_relations(synset_name: str) -> Dict[str, Any]:
    """Extract in-depth lexical and semantic relations for a specific WordNet synset."""
    try:
        syn = wn.synset(synset_name)
    except Exception:
        return {}

    # Hypernyms and taxonomic path to root
    hypernyms = syn.hypernyms()
    hypernym_paths = syn.hypernym_paths()
    longest_path = max(hypernym_paths, key=len) if hypernym_paths else []
    path_names = [h.name().split(".")[0].replace("_", " ") for h in longest_path]

    # Subordinate terms (Hyponyms)
    hyponyms = syn.hyponyms()
    
    # Meronyms (Part, Member, Substance)
    part_meronyms = syn.part_meronyms()
    member_meronyms = syn.member_meronyms()
    substance_meronyms = syn.substance_meronyms()

    # Holonyms (Whole)
    part_holonyms = syn.part_holonyms()
    member_holonyms = syn.member_holonyms()
    substance_holonyms = syn.substance_holonyms()

    # Antonyms & Derivational relations from lemmas
    antonyms = []
    derivations = []
    for l in syn.lemmas():
        for ant in l.antonyms():
            antonyms.append(f"{l.name()} ↔ {ant.name()}")
        for der in l.derivationally_related_forms():
            derivations.append(f"{l.name()} ({syn.pos()}) → {der.name()} ({der.synset().pos()})")

    # Sister / Coordinate terms (sharing same hypernym)
    coordinate_terms = set()
    for h in hypernyms:
        for sibling in h.hyponyms():
            if sibling != syn:
                coordinate_terms.add(sibling.name().split(".")[0].replace("_", " "))

    # Verb-specific relations
    entailments = syn.entailments() if hasattr(syn, 'entailments') else []
    causes = syn.causes() if hasattr(syn, 'causes') else []

    return {
        "synset": syn,
        "name": syn.name(),
        "pos": syn.pos(),
        "definition": syn.definition(),
        "examples": syn.examples(),
        "lemmas": [l.name().replace("_", " ") for l in syn.lemmas()],
        "hypernyms": [h.name() for h in hypernyms],
        "taxonomy_depth": syn.min_depth(),
        "taxonomy_path": path_names,
        "hyponyms": [h.name() for h in hyponyms],
        "part_meronyms": [m.name() for m in part_meronyms],
        "member_meronyms": [m.name() for m in member_meronyms],
        "substance_meronyms": [m.name() for m in substance_meronyms],
        "part_holonyms": [h.name() for h in part_holonyms],
        "member_holonyms": [h.name() for h in member_holonyms],
        "substance_holonyms": [h.name() for h in substance_holonyms],
        "antonyms": list(set(antonyms)),
        "derivations": list(set(derivations)),
        "coordinate_terms": sorted(list(coordinate_terms))[:15],
        "entailments": [e.name() for e in entailments],
        "causes": [c.name() for c in causes]
    }


def build_lexical_network_graph(synset_name: str, max_hyponyms: int = 8) -> go.Figure:
    """
    Construct an interactive Plotly 2D Network Graph showing the target synset
    and its lexical relations (Hypernyms, Hyponyms, Meronyms, Sister terms, Antonyms).
    """
    try:
        syn = wn.synset(synset_name)
    except Exception:
        fig = go.Figure()
        fig.update_layout(title="Synset not found")
        return fig

    G = nx.DiGraph()
    center_label = f"🎯 {syn.name()}"
    G.add_node(center_label, node_type="Center", definition=syn.definition(), color="#6366F1", size=28)

    # 1. Hypernyms (Broader concepts / Is-A)
    for h in syn.hypernyms():
        h_label = f"⬆️ {h.name()}"
        G.add_node(h_label, node_type="Hypernym (Is-A)", definition=h.definition(), color="#3B82F6", size=20)
        G.add_edge(center_label, h_label, rel="is-a (hypernym)")

    # 2. Hyponyms (Narrower concepts / Subtypes)
    for h in syn.hyponyms()[:max_hyponyms]:
        hypo_label = f"⬇️ {h.name()}"
        G.add_node(hypo_label, node_type="Hyponym (Subtype)", definition=h.definition(), color="#10B981", size=18)
        G.add_edge(hypo_label, center_label, rel="subtype-of (hyponym)")

    # 3. Meronyms (Part-of)
    for m in (syn.part_meronyms() + syn.member_meronyms())[:5]:
        m_label = f"🧩 {m.name()}"
        G.add_node(m_label, node_type="Meronym (Part)", definition=m.definition(), color="#F59E0B", size=18)
        G.add_edge(m_label, center_label, rel="part-of (meronym)")

    # 4. Antonyms
    for l in syn.lemmas():
        for ant in l.antonyms():
            ant_label = f"⚡ {ant.name()} ({ant.synset().name()})"
            G.add_node(ant_label, node_type="Antonym (Opposite)", definition=ant.synset().definition(), color="#EF4444", size=18)
            G.add_edge(center_label, ant_label, rel="antonym-of")

    # 5. Coordinate terms (Sister terms)
    for h in syn.hypernyms()[:1]:
        for sib in h.hyponyms():
            if sib != syn and len(G.nodes) < 22:
                sib_label = f"↔️ {sib.name()}"
                G.add_node(sib_label, node_type="Coordinate (Sister)", definition=sib.definition(), color="#8B5CF6", size=16)
                G.add_edge(sib_label, f"⬆️ {h.name()}", rel="co-hyponym")

    # Compute graph positions
    pos = nx.spring_layout(G, k=0.8, seed=42)

    # Prepare edge traces
    edge_x = []
    edge_y = []
    edge_text = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_text.append(edge[2].get("rel", ""))

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.5, color="#94A3B8"),
        hoverinfo="none",
        mode="lines"
    )

    # Prepare node traces
    node_x = []
    node_y = []
    node_colors = []
    node_sizes = []
    node_hover = []
    node_text = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        data = G.nodes[node]
        node_colors.append(data.get("color", "#64748B"))
        node_sizes.append(data.get("size", 18))
        node_text.append(node.split()[0] + " " + node.split()[1] if len(node.split()) > 1 else node)
        hover_info = f"<b>{node}</b><br>Type: {data.get('node_type')}<br>Def: {data.get('definition')}"
        node_hover.append(hover_info)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        hoverinfo="text",
        text=node_text,
        textposition="top center",
        textfont=dict(size=11, color="#1E293B"),
        hovertext=node_hover,
        marker=dict(
            showscale=False,
            color=node_colors,
            size=node_sizes,
            line=dict(width=2, color="#FFFFFF")
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=f"Lexical Relations Network: <i>{synset_name}</i>",
                        titlefont_size=16,
                        showlegend=False,
                        hovermode="closest",
                        margin=dict(b=20, l=10, r=10, t=50),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        plot_bgcolor="#F8FAFC",
                        paper_bgcolor="#FFFFFF"
                    ))
    return fig
