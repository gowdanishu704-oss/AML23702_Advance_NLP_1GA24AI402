"""
Text-to-Cypher and Natural Language Query Intent Router.
Combines deterministic template matching for canonical queries with
dynamic LLM Cypher generation for complex graph questions.
Includes Cypher safety guardrails to ensure read-only execution.
"""

import re
from typing import Dict, Any, Optional, Tuple, List
from backend.query_templates import QUERY_TEMPLATES
from data.seed_data import MOVIES, PERSONS

# Known movie titles and person names for high-precision entity extraction
KNOWN_MOVIES = [m["title"] for m in MOVIES]
KNOWN_PERSONS = [p["name"] for p in PERSONS]

# Forbidden mutating keywords in Cypher queries for security
FORBIDDEN_KEYWORDS = [
    "CREATE", "DELETE", "DETACH", "SET", "REMOVE", "DROP",
    "MERGE", "CALL apoc.periodic", "APOC.TRIGGER", "ALTER", "GRANT"
]

SCHEMA_CONTEXT = """
Neo4j Movies Graph Schema:
- Node Labels:
  * (:Movie {title: STRING, released: INTEGER, tagline: STRING})
  * (:Person {name: STRING, born: INTEGER})
- Relationship Types:
  * (:Person)-[:ACTED_IN {roles: LIST OF STRING}]->(:Movie)
  * (:Person)-[:DIRECTED]->(:Movie)
  * (:Person)-[:PRODUCED]->(:Movie)
  * (:Person)-[:WROTE]->(:Movie)
  * (:Person)-[:REVIEWED {rating: INTEGER, summary: STRING}]->(:Movie)
  * (:Person)-[:FOLLOWS]->(:Person)
"""


def extract_entities(text: str) -> Dict[str, List[str]]:
    """
    Extract recognized Movie titles and Person names from natural language text.
    Uses case-insensitive matching against known entities in the Movies graph.
    """
    found_movies = []
    found_persons = []
    text_lower = text.lower()

    for m in KNOWN_MOVIES:
        # Match whole word / title boundaries
        pattern = r"\b" + re.escape(m.lower()) + r"\b"
        if re.search(pattern, text_lower):
            found_movies.append(m)

    for p in KNOWN_PERSONS:
        pattern = r"\b" + re.escape(p.lower()) + r"\b"
        if re.search(pattern, text_lower):
            found_persons.append(p)

    # Sort longer names first to avoid partial matches
    found_movies = sorted(list(set(found_movies)), key=len, reverse=True)
    found_persons = sorted(list(set(found_persons)), key=len, reverse=True)

    return {"movies": found_movies, "persons": found_persons}


def match_template(question: str) -> Optional[Dict[str, Any]]:
    """
    Deterministic rule-based intent router that maps natural language queries
    to pre-validated Cypher query templates.
    """
    q_lower = question.lower().strip()
    entities = extract_entities(question)
    movies = entities["movies"]
    persons = entities["persons"]

    # 1. Recommendation intents
    if any(k in q_lower for k in ["recommend", "suggestion", "similar to", "fans of", "what should i watch"]):
        if persons:
            return {
                "template_id": "RECOMMEND_BY_ACTOR",
                "params": {"actor": persons[0]},
                "matched_type": "deterministic",
                "explanation": f"Matched Collaborative Recommendation for fan of '{persons[0]}'"
            }
        elif movies:
            return {
                "template_id": "MOVIE_CAST_CREW",
                "params": {"title": movies[0]},
                "matched_type": "deterministic",
                "explanation": f"Retrieved metadata to generate recommendations based on '{movies[0]}'"
            }

    # 2. Connection / Path intents (Kevin Bacon game, shortest path)
    if any(k in q_lower for k in ["how are", "connected", "path between", "bacon number", "degrees of separation"]):
        if len(persons) >= 2:
            return {
                "template_id": "SHORTEST_PATH",
                "params": {"person1": persons[0], "person2": persons[1]},
                "matched_type": "deterministic",
                "explanation": f"Matched Shortest Path between '{persons[0]}' and '{persons[1]}'"
            }

    # 3. Shared movies intent
    if any(k in q_lower for k in ["together", "shared", "both act", "co-star in"]):
        if len(persons) >= 2:
            return {
                "template_id": "SHARED_MOVIES",
                "params": {"person1": persons[0], "person2": persons[1]},
                "matched_type": "deterministic",
                "explanation": f"Matched Shared Movies for '{persons[0]}' and '{persons[1]}'"
            }

    # 4. Co-actors intent
    if any(k in q_lower for k in ["co-star", "costar", "co-actor", "who worked with", "who acted with"]):
        if persons:
            return {
                "template_id": "CO_ACTORS",
                "params": {"actor": persons[0]},
                "matched_type": "deterministic",
                "explanation": f"Matched Co-Actor Network for '{persons[0]}'"
            }

    # 5. Director intent
    if any(k in q_lower for k in ["directed", "director", "who directed"]):
        if movies and not persons:
            return {
                "template_id": "MOVIE_CAST_CREW",
                "params": {"title": movies[0]},
                "matched_type": "deterministic",
                "explanation": f"Matched Movie Cast & Crew for '{movies[0]}'"
            }
        elif persons:
            return {
                "template_id": "DIRECTOR_MOVIES",
                "params": {"director": persons[0]},
                "matched_type": "deterministic",
                "explanation": f"Matched Director Movies for '{persons[0]}'"
            }

    # 6. Reviews / Top rated intent
    if any(k in q_lower for k in ["top rated", "best rated", "reviews", "highest rated", "rating"]):
        return {
            "template_id": "TOP_MOVIES_BY_REVIEWS",
            "params": {},
            "matched_type": "deterministic",
            "explanation": "Matched Top Movies by User Reviews"
        }

    # 7. Multi-talented (acted and directed)
    if any(k in q_lower for k in ["both act and direct", "acted and directed", "director and actor", "double threat"]):
        return {
            "template_id": "MULTITALENTED_PERSONS",
            "params": {},
            "matched_type": "deterministic",
            "explanation": "Matched Actor-Directors query"
        }

    # 8. Movie Details / Cast
    if movies:
        return {
            "template_id": "MOVIE_CAST_CREW",
            "params": {"title": movies[0]},
            "matched_type": "deterministic",
            "explanation": f"Matched Movie Cast & Crew for '{movies[0]}'"
        }

    # 9. Actor Filmography
    if persons:
        return {
            "template_id": "ACTOR_FILMOGRAPHY",
            "params": {"actor": persons[0]},
            "matched_type": "deterministic",
            "explanation": f"Matched Actor Filmography for '{persons[0]}'"
        }

    return None


def validate_and_sanitize_cypher(cypher: str) -> Tuple[bool, str]:
    """
    Validate that generated Cypher queries are strictly read-only and safe to execute.
    """
    clean_cypher = cypher.strip().rstrip(";")
    upper_cypher = clean_cypher.upper()

    for kw in FORBIDDEN_KEYWORDS:
        # Check for standalone keyword match
        if re.search(r"\b" + kw + r"\b", upper_cypher):
            return False, f"Query rejected: contains forbidden mutating keyword '{kw}'"

    if not upper_cypher.startswith("MATCH") and not upper_cypher.startswith("RETURN"):
        return False, "Query rejected: read queries must start with MATCH or RETURN"

    return True, clean_cypher


def generate_cypher_prompt(question: str) -> str:
    """
    Construct a few-shot prompt for an LLM to generate Cypher based on the schema.
    """
    return f"""You are a Neo4j Cypher expert for a Movie Knowledge Graph.
Convert the following user question into a valid, safe, read-only Cypher query.

{SCHEMA_CONTEXT}

Rules:
1. Return ONLY the Cypher query. No explanations, no markdown formatting, no backticks.
2. Only use MATCH, WHERE, RETURN, ORDER BY, LIMIT.
3. NEVER use mutating operations (CREATE, DELETE, SET, etc.).
4. Use case-insensitive string matching: `toLower(n.name) = toLower('...')` or `toLower(n.title) CONTAINS toLower('...')`.
5. Limit results to 15 unless specified otherwise.

Few-shot Examples:
Q: What movies did Keanu Reeves act in?
Cypher: MATCH (p:Person {{name: 'Keanu Reeves'}})-[r:ACTED_IN]->(m:Movie) RETURN m.title AS movie, m.released AS released, r.roles AS roles ORDER BY m.released DESC

Q: Who directed The Matrix?
Cypher: MATCH (p:Person)-[:DIRECTED]->(m:Movie {{title: 'The Matrix'}}) RETURN p.name AS director, p.born AS born

Q: Which actors were born after 1970?
Cypher: MATCH (p:Person)-[:ACTED_IN]->(m:Movie) WHERE p.born > 1970 RETURN DISTINCT p.name AS actor, p.born AS born ORDER BY p.born ASC LIMIT 10

User Question: {question}
Cypher:"""
