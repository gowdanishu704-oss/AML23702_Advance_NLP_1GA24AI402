"""
Curated Set of Cypher Query Templates for the Neo4j Movies Knowledge Graph.
Contains parameterized Cypher templates for Movie QA, Graph-Based Recommendations,
Cast & Crew Retrieval, Network Analysis, and Path Traversal.
"""

from typing import Dict, Any, List

QUERY_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "ACTOR_FILMOGRAPHY": {
        "id": "ACTOR_FILMOGRAPHY",
        "title": "Actor Filmography & Roles",
        "category": "QA",
        "description": "Finds all movies an actor has starred in, their specific character roles, and release years.",
        "sample_question": "What movies has Keanu Reeves acted in and what were his roles?",
        "params": {"actor": "Keanu Reeves"},
        "cypher": """
MATCH (p:Person)-[r:ACTED_IN]->(m:Movie)
WHERE toLower(p.name) = toLower($actor)
RETURN p.name AS actor, m.title AS movie, m.released AS released, m.tagline AS tagline, r.roles AS roles
ORDER BY m.released DESC
"""
    },

    "DIRECTOR_MOVIES": {
        "id": "DIRECTOR_MOVIES",
        "title": "Director's Filmography",
        "category": "QA",
        "description": "Finds all movies directed by a given director with release year and tagline.",
        "sample_question": "Which movies were directed by Christopher Nolan or the Wachowskis?",
        "params": {"director": "Lilly Wachowski"},
        "cypher": """
MATCH (p:Person)-[:DIRECTED]->(m:Movie)
WHERE toLower(p.name) = toLower($director)
RETURN p.name AS director, m.title AS movie, m.released AS released, m.tagline AS tagline
ORDER BY m.released DESC
"""
    },

    "MOVIE_CAST_CREW": {
        "id": "MOVIE_CAST_CREW",
        "title": "Movie Cast, Crew & Metadata",
        "category": "QA",
        "description": "Retrieves the complete cast and crew (actors, directors, producers, writers) and movie details.",
        "sample_question": "Who starred in and directed The Matrix?",
        "params": {"title": "The Matrix"},
        "cypher": """
MATCH (p:Person)-[r]->(m:Movie)
WHERE toLower(m.title) = toLower($title)
RETURN p.name AS person, type(r) AS relationship, r.roles AS roles, p.born AS born, m.title AS movie, m.released AS released, m.tagline AS tagline
ORDER BY type(r), p.name
"""
    },

    "CO_ACTORS": {
        "id": "CO_ACTORS",
        "title": "Co-Actors Collaboration Network",
        "category": "Network Analysis",
        "description": "Finds all actors who have shared the screen with a specific actor, along with shared movies and frequency.",
        "sample_question": "Who has Tom Hanks co-starred with the most?",
        "params": {"actor": "Tom Hanks"},
        "cypher": """
MATCH (p1:Person)-[:ACTED_IN]->(m:Movie)<-[:ACTED_IN]-(p2:Person)
WHERE toLower(p1.name) = toLower($actor) AND p1 <> p2
RETURN p2.name AS name, count(DISTINCT m) AS count, collect(DISTINCT m.title) AS shared_movies
ORDER BY count DESC, p2.name ASC
LIMIT 10
"""
    },

    "RECOMMEND_BY_ACTOR": {
        "id": "RECOMMEND_BY_ACTOR",
        "title": "Collaborative Graph Movie Recommendation",
        "category": "Recommendations",
        "description": "Recommends movies using 2-hop co-actor collaborative filtering: movies acted in by frequent co-stars that the user's actor was not in.",
        "sample_question": "Can you recommend movies for a fan of Keanu Reeves?",
        "params": {"actor": "Keanu Reeves"},
        "cypher": """
MATCH (target:Person)-[:ACTED_IN]->(m:Movie)<-[:ACTED_IN]-(coStar:Person)
WHERE toLower(target.name) = toLower($actor)
MATCH (coStar)-[:ACTED_IN]->(recMovie:Movie)
WHERE NOT (target)-[:ACTED_IN]->(recMovie)
RETURN recMovie.title AS movie, recMovie.released AS released, recMovie.tagline AS tagline,
       count(DISTINCT coStar) AS score, collect(DISTINCT coStar.name) AS via_co_actors
ORDER BY score DESC, recMovie.released DESC
LIMIT 5
"""
    },

    "RECOMMEND_BY_DIRECTOR_STYLE": {
        "id": "RECOMMEND_BY_DIRECTOR_STYLE",
        "title": "Director Style Recommendations",
        "category": "Recommendations",
        "description": "Recommends other movies directed by the same directors who directed movies starring the given actor.",
        "sample_question": "Recommend movies based on the directors who worked with Tom Cruise.",
        "params": {"actor": "Tom Cruise"},
        "cypher": """
MATCH (a:Person)-[:ACTED_IN]->(m1:Movie)<-[:DIRECTED]-(d:Person)
WHERE toLower(a.name) = toLower($actor)
MATCH (d)-[:DIRECTED]->(m2:Movie)
WHERE m1 <> m2 AND NOT (a)-[:ACTED_IN]->(m2)
RETURN m2.title AS movie, m2.released AS released, m2.tagline AS tagline, d.name AS director
ORDER BY m2.released DESC
LIMIT 5
"""
    },

    "SHORTEST_PATH": {
        "id": "SHORTEST_PATH",
        "title": "Degrees of Separation (Shortest Path)",
        "category": "Network Analysis",
        "description": "Finds the shortest relationship path between two people (e.g. Kevin Bacon connection or Six Degrees).",
        "sample_question": "How are Kevin Bacon and Keanu Reeves connected in the movie graph?",
        "params": {"person1": "Kevin Bacon", "person2": "Keanu Reeves"},
        "cypher": """
MATCH (p1:Person), (p2:Person)
WHERE toLower(p1.name) = toLower($person1) AND toLower(p2.name) = toLower($person2)
MATCH path = shortestPath((p1)-[*]-(p2))
RETURN [n in nodes(path) | coalesce(n.name, n.title)] AS path, length(path) AS length
"""
    },

    "SHARED_MOVIES": {
        "id": "SHARED_MOVIES",
        "title": "Shared Movies Between Two Actors",
        "category": "QA",
        "description": "Finds all movies in which both specified individuals participated.",
        "sample_question": "Did Tom Cruise and Jack Nicholson act in any movie together?",
        "params": {"person1": "Tom Cruise", "person2": "Jack Nicholson"},
        "cypher": """
MATCH (p1:Person)-[:ACTED_IN]->(m:Movie)<-[:ACTED_IN]-(p2:Person)
WHERE toLower(p1.name) = toLower($person1) AND toLower(p2.name) = toLower($person2)
RETURN m.title AS movie, m.released AS released, m.tagline AS tagline
"""
    },

    "TOP_MOVIES_BY_REVIEWS": {
        "id": "TOP_MOVIES_BY_REVIEWS",
        "title": "Top Reviewed Movies",
        "category": "Aggregations",
        "description": "Finds movies with the highest ratings and user reviews in the knowledge graph.",
        "sample_question": "Which movies have the best ratings and reviews?",
        "params": {},
        "cypher": """
MATCH (p:Person)-[r:REVIEWED]->(m:Movie)
RETURN m.title AS movie, m.released AS released, round(avg(r.rating), 1) AS avg_rating, count(r) AS review_count, collect(r.summary) AS sample_reviews
ORDER BY avg_rating DESC, review_count DESC
LIMIT 5
"""
    },

    "MULTITALENTED_PERSONS": {
        "id": "MULTITALENTED_PERSONS",
        "title": "Actor-Directors (Double Threats)",
        "category": "Aggregations",
        "description": "Finds persons who have both acted in and directed movies in the knowledge graph.",
        "sample_question": "Which people in the database have both acted in and directed movies?",
        "params": {},
        "cypher": """
MATCH (p:Person)-[:ACTED_IN]->(m1:Movie)
MATCH (p)-[:DIRECTED]->(m2:Movie)
RETURN p.name AS person, p.born AS born, collect(DISTINCT m1.title) AS acted_in, collect(DISTINCT m2.title) AS directed
ORDER BY size(acted_in) + size(directed) DESC
LIMIT 10
"""
    },

    "PROLIFIC_ACTORS": {
        "id": "PROLIFIC_ACTORS",
        "title": "Most Prolific Actors",
        "category": "Aggregations",
        "description": "Ranks actors by the number of movies they appear in.",
        "sample_question": "Who are the top 5 most prolific actors in the database?",
        "params": {"limit": 5},
        "cypher": """
MATCH (p:Person)-[:ACTED_IN]->(m:Movie)
RETURN p.name AS actor, count(m) AS movie_count, collect(m.title) AS movies
ORDER BY movie_count DESC
LIMIT $limit
"""
    }
}


def get_template(template_id: str) -> Dict[str, Any]:
    """Retrieve template definition by ID."""
    return QUERY_TEMPLATES.get(template_id, QUERY_TEMPLATES["ACTOR_FILMOGRAPHY"])


def list_templates() -> List[Dict[str, Any]]:
    """Return all available query templates."""
    return list(QUERY_TEMPLATES.values())
