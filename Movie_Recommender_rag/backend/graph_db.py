"""
Neo4j Graph Database Driver & In-Memory Graph Engine Fallback.
Provides unified query execution, connection health checks, schema introspection,
and automatic fallback to an embedded graph populated with the Neo4j Movies dataset.
"""

import os
import re
from typing import List, Dict, Any, Optional, Tuple
import networkx as nx
from neo4j import GraphDatabase, Driver
from data.seed_data import MOVIES, PERSONS, RELATIONSHIPS


class InMemoryMovieGraph:
    """
    In-memory graph simulator loaded with the canonical Neo4j Movies dataset.
    Used when Neo4j Desktop or AuraDB is offline or credentials are not yet provided.
    Evaluates Cypher-like queries and template queries with exact graph semantics.
    """
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.movies: Dict[str, Dict[str, Any]] = {}
        self.persons: Dict[str, Dict[str, Any]] = {}
        self._load_seed_data()

    def _load_seed_data(self):
        self.graph.clear()
        for movie in MOVIES:
            title = movie["title"]
            self.movies[title] = movie
            self.graph.add_node(title, label="Movie", **movie)

        for person in PERSONS:
            name = person["name"]
            self.persons[name] = person
            self.graph.add_node(name, label="Person", **person)

        for rel in RELATIONSHIPS:
            src = rel["source"]
            tgt = rel["target"]
            rel_type = rel["type"]
            props = rel.get("props", {})
            self.graph.add_edge(src, tgt, type=rel_type, **props)

    def get_schema_summary(self) -> Dict[str, Any]:
        return {
            "node_count": self.graph.number_of_nodes(),
            "relationship_count": self.graph.number_of_edges(),
            "movie_count": len(self.movies),
            "person_count": len(self.persons),
            "labels": ["Movie", "Person"],
            "relationship_types": ["ACTED_IN", "DIRECTED", "PRODUCED", "WROTE", "REVIEWED", "FOLLOWS"],
            "backend": "Embedded In-Memory Graph (Neo4j Movies Dataset)"
        }

    def execute_template(self, template_key: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Execute high-level graph query patterns directly against the embedded graph.
        """
        results = []
        p_raw = params.get("actor") or params.get("person") or params.get("name")
        m_raw = params.get("title") or params.get("movie")

        # Case-insensitive resolution to exact node key in graph
        p_name = None
        if p_raw:
            for p in self.persons:
                if p.lower() == p_raw.lower():
                    p_name = p
                    break
            if not p_name and p_raw in self.graph:
                p_name = p_raw

        m_title = None
        if m_raw:
            for m in self.movies:
                if m.lower() == m_raw.lower():
                    m_title = m
                    break
            if not m_title and m_raw in self.graph:
                m_title = m_raw

        if template_key in ["ACTOR_MOVIES", "ACTOR_FILMOGRAPHY"] and p_name:
            for u, v, data in self.graph.out_edges(p_name, data=True):
                if data.get("type") == "ACTED_IN":
                    movie_node = self.graph.nodes.get(v, {})
                    results.append({
                        "actor": p_name,
                        "movie": v,
                        "released": movie_node.get("released"),
                        "tagline": movie_node.get("tagline"),
                        "roles": data.get("roles", [])
                    })

        elif template_key == "DIRECTOR_MOVIES" and p_name:
            for u, v, data in self.graph.out_edges(p_name, data=True):
                if data.get("type") == "DIRECTED":
                    movie_node = self.graph.nodes.get(v, {})
                    results.append({
                        "director": p_name,
                        "movie": v,
                        "released": movie_node.get("released"),
                        "tagline": movie_node.get("tagline")
                    })

        elif template_key == "MOVIE_CAST_CREW" and m_title:
            for u, v, data in self.graph.in_edges(m_title, data=True):
                person_node = self.graph.nodes.get(u, {})
                results.append({
                    "person": u,
                    "born": person_node.get("born"),
                    "relationship": data.get("type"),
                    "roles": data.get("roles", []),
                    "movie": m_title,
                    "released": self.graph.nodes[m_title].get("released"),
                    "tagline": self.graph.nodes[m_title].get("tagline")
                })

        elif template_key == "CO_ACTORS" and p_name:
            actor_movies = set()
            for _, v, data in self.graph.out_edges(p_name, data=True):
                if data.get("type") == "ACTED_IN":
                    actor_movies.add(v)
            
            co_actors = {}
            for movie in actor_movies:
                for u, _, data in self.graph.in_edges(movie, data=True):
                    if data.get("type") == "ACTED_IN" and u != p_name:
                        if u not in co_actors:
                            co_actors[u] = {"name": u, "shared_movies": [], "count": 0}
                        co_actors[u]["shared_movies"].append(movie)
                        co_actors[u]["count"] += 1
            
            results = sorted(list(co_actors.values()), key=lambda x: x["count"], reverse=True)

        elif template_key == "RECOMMEND_BY_ACTOR" and p_name:
            co_actors = set()
            actor_movies = set()
            for _, m, data in self.graph.out_edges(p_name, data=True):
                if data.get("type") == "ACTED_IN":
                    actor_movies.add(m)
            
            for m in actor_movies:
                for other_p, _, data in self.graph.in_edges(m, data=True):
                    if data.get("type") == "ACTED_IN" and other_p != p_name:
                        co_actors.add(other_p)
            
            rec_movies: Dict[str, Dict[str, Any]] = {}
            for co_p in co_actors:
                for _, rec_m, data in self.graph.out_edges(co_p, data=True):
                    if data.get("type") == "ACTED_IN" and rec_m not in actor_movies:
                        if rec_m not in rec_movies:
                            rec_movies[rec_m] = {
                                "movie": rec_m,
                                "released": self.graph.nodes[rec_m].get("released"),
                                "tagline": self.graph.nodes[rec_m].get("tagline"),
                                "via_co_actors": [],
                                "score": 0
                            }
                        if co_p not in rec_movies[rec_m]["via_co_actors"]:
                            rec_movies[rec_m]["via_co_actors"].append(co_p)
                            rec_movies[rec_m]["score"] += 1
            
            results = sorted(list(rec_movies.values()), key=lambda x: x["score"], reverse=True)[:6]

        elif template_key == "RECOMMEND_BY_DIRECTOR_STYLE" and p_name:
            actor_movies = set()
            for _, m, data in self.graph.out_edges(p_name, data=True):
                if data.get("type") == "ACTED_IN":
                    actor_movies.add(m)
            directors = set()
            for m in actor_movies:
                for u, _, data in self.graph.in_edges(m, data=True):
                    if data.get("type") == "DIRECTED":
                        directors.add(u)
            rec_movies = []
            for d in directors:
                for _, m, data in self.graph.out_edges(d, data=True):
                    if data.get("type") == "DIRECTED" and m not in actor_movies:
                        rec_movies.append({
                            "movie": m,
                            "released": self.graph.nodes[m].get("released"),
                            "tagline": self.graph.nodes[m].get("tagline"),
                            "director": d
                        })
            results = rec_movies[:5]

        elif template_key == "SHARED_MOVIES":
            p1_raw = params.get("person1") or ""
            p2_raw = params.get("person2") or ""
            p1 = next((p for p in self.persons if p.lower() == p1_raw.lower()), p1_raw)
            p2 = next((p for p in self.persons if p.lower() == p2_raw.lower()), p2_raw)
            m1 = {v for _, v, d in self.graph.out_edges(p1, data=True) if d.get("type") == "ACTED_IN"}
            m2 = {v for _, v, d in self.graph.out_edges(p2, data=True) if d.get("type") == "ACTED_IN"}
            shared = m1.intersection(m2)
            for m in shared:
                results.append({
                    "movie": m,
                    "released": self.graph.nodes[m].get("released"),
                    "tagline": self.graph.nodes[m].get("tagline")
                })

        elif template_key == "SHORTEST_PATH":
            p1_raw = params.get("person1") or params.get("source") or ""
            p2_raw = params.get("person2") or params.get("target") or ""
            p1 = next((p for p in self.persons if p.lower() == p1_raw.lower()), p1_raw)
            p2 = next((p for p in self.persons if p.lower() == p2_raw.lower()), p2_raw)
            if p1 and p2 and p1 in self.graph and p2 in self.graph:
                undirected = self.graph.to_undirected()
                try:
                    path = nx.shortest_path(undirected, source=p1, target=p2)
                    results.append({
                        "source": p1,
                        "target": p2,
                        "path": path,
                        "degrees_of_separation": (len(path) - 1) // 2
                    })
                except nx.NetworkXNoPath:
                    results.append({"source": p1, "target": p2, "error": "No connection found"})

        elif template_key == "TOP_MOVIES_BY_REVIEWS":
            movie_reviews: Dict[str, List[int]] = {}
            for u, v, data in self.graph.edges(data=True):
                if data.get("type") == "REVIEWED" and "rating" in data:
                    movie_reviews.setdefault(v, []).append(data["rating"])
            for m, ratings in movie_reviews.items():
                results.append({
                    "movie": m,
                    "avg_rating": round(sum(ratings) / len(ratings), 1),
                    "review_count": len(ratings),
                    "released": self.graph.nodes[m].get("released")
                })
            results = sorted(results, key=lambda x: x["avg_rating"], reverse=True)

        elif template_key == "MULTITALENTED_PERSONS":
            for p in self.persons:
                acted = [v for _, v, d in self.graph.out_edges(p, data=True) if d.get("type") == "ACTED_IN"]
                directed = [v for _, v, d in self.graph.out_edges(p, data=True) if d.get("type") == "DIRECTED"]
                if acted and directed:
                    results.append({
                        "person": p,
                        "born": self.persons[p].get("born"),
                        "acted_in": acted,
                        "directed": directed
                    })
            results = sorted(results, key=lambda x: len(x["acted_in"]) + len(x["directed"]), reverse=True)

        elif template_key == "PROLIFIC_ACTORS":
            actor_counts = []
            for p in self.persons:
                acted = [v for _, v, d in self.graph.out_edges(p, data=True) if d.get("type") == "ACTED_IN"]
                if acted:
                    actor_counts.append({
                        "actor": p,
                        "movie_count": len(acted),
                        "movies": acted
                    })
            results = sorted(actor_counts, key=lambda x: x["movie_count"], reverse=True)[:params.get("limit", 10)]

        elif template_key == "ALL_MOVIES":
            limit = params.get("limit", 25)
            for m in list(self.movies.values())[:limit]:
                results.append(m)

        elif template_key == "ALL_ACTORS":
            limit = params.get("limit", 25)
            for p in list(self.persons.values())[:limit]:
                results.append(p)

        return results

    def query_freeform(self, cypher: str) -> List[Dict[str, Any]]:
        """
        Lightweight Cypher evaluator for simple MATCH clauses when running offline.
        Handles MATCH (p:Person {name: '...'}), MATCH (m:Movie {title: '...'}), etc.
        """
        cypher_clean = cypher.strip().replace("\n", " ")
        results = []

        # Case 1: Person name match
        match_person = re.search(r"Person\s*\{\s*name:\s*['\"]([^'\"]+)['\"]\s*\}", cypher_clean, re.IGNORECASE)
        match_movie = re.search(r"Movie\s*\{\s*title:\s*['\"]([^'\"]+)['\"]\s*\}", cypher_clean, re.IGNORECASE)

        if match_person and "ACTED_IN" in cypher_clean.upper():
            return self.execute_template("ACTOR_MOVIES", {"actor": match_person.group(1)})
        if match_person and "DIRECTED" in cypher_clean.upper():
            return self.execute_template("DIRECTOR_MOVIES", {"director": match_person.group(1)})
        if match_movie:
            return self.execute_template("MOVIE_CAST_CREW", {"title": match_movie.group(1)})
        
        # General scan fallback
        return self.execute_template("ALL_MOVIES", {"limit": 10})


class Neo4jGraphService:
    """
    Unified Graph Service that manages:
    - Live Neo4j connection (Neo4j Desktop / Neo4j AuraDB / Docker)
    - Automatic fallback to InMemoryMovieGraph when offline
    - Seeding and migration from data/movies_dataset.cql
    - Parameterized Cypher execution and schema introspection
    """
    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: str = "neo4j"
    ):
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password")
        self.database = database or os.getenv("NEO4J_DATABASE", "neo4j")
        
        self.driver: Optional[Driver] = None
        self.is_connected = False
        self.fallback_engine = InMemoryMovieGraph()
        
        # Try to connect
        self.connect()

    def connect(self) -> Tuple[bool, str]:
        """Attempt connection to Neo4j database."""
        if not self.uri or not self.password:
            self.is_connected = False
            return False, "Neo4j URI or password not configured. Using embedded graph fallback."
        
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
                connection_timeout=3.0,
                max_connection_lifetime=60
            )
            self.driver.verify_connectivity()
            self.is_connected = True
            return True, f"Successfully connected to Neo4j at {self.uri}"
        except Exception as e:
            self.is_connected = False
            self.driver = None
            return False, f"Could not connect to Neo4j ({str(e)}). Active mode: Embedded In-Memory Movies Graph."

    def close(self):
        if self.driver:
            self.driver.close()

    def get_status(self) -> Dict[str, Any]:
        """Return connectivity status and active backend."""
        if self.is_connected:
            return {
                "status": "connected",
                "backend": "Neo4j (Live)",
                "uri": self.uri,
                "database": self.database
            }
        return {
            "status": "embedded_fallback",
            "backend": "Embedded In-Memory Graph (Neo4j Movies Dataset)",
            "uri": "local://memory",
            "database": "embedded"
        }

    def get_schema(self) -> Dict[str, Any]:
        """Introspect node labels, relationship types, and counts."""
        if not self.is_connected:
            return self.fallback_engine.get_schema_summary()

        try:
            with self.driver.session(database=self.database) as session:
                # Node counts
                nodes_res = session.run("MATCH (n) RETURN labels(n) AS labels, count(n) AS count").data()
                # Rel counts
                rels_res = session.run("MATCH ()-[r]->() RETURN type(r) AS rel_type, count(r) AS count").data()
                
                total_nodes = sum(r["count"] for r in nodes_res)
                total_rels = sum(r["count"] for r in rels_res)
                
                return {
                    "node_count": total_nodes,
                    "relationship_count": total_rels,
                    "node_distribution": nodes_res,
                    "relationship_distribution": rels_res,
                    "labels": ["Movie", "Person"],
                    "relationship_types": [r["rel_type"] for r in rels_res],
                    "backend": f"Neo4j Live Instance ({self.uri})"
                }
        except Exception as e:
            fallback = self.fallback_engine.get_schema_summary()
            fallback["warning"] = f"Failed to fetch live schema ({str(e)}), used fallback."
            return fallback

    def seed_database(self, cql_path: Optional[str] = None) -> Tuple[bool, str]:
        """
        Seed the connected Neo4j instance with data/movies_dataset.cql.
        """
        if not self.is_connected:
            return False, "Cannot seed: Neo4j is not connected. Embedded fallback already contains the dataset."
        
        path = cql_path or os.path.join(os.path.dirname(__file__), "..", "data", "movies_dataset.cql")
        if not os.path.exists(path):
            return False, f"CQL file not found at: {path}"

        try:
            with open(path, "r", encoding="utf-8") as f:
                cql_content = f.read()

            # Split statements by semicolon or execute blocks
            statements = [stmt.strip() for stmt in cql_content.split(";") if stmt.strip() and not stmt.strip().startswith("//")]

            with self.driver.session(database=self.database) as session:
                # First create constraint / index
                try:
                    session.run("CREATE CONSTRAINT person_name IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE")
                    session.run("CREATE CONSTRAINT movie_title IF NOT EXISTS FOR (m:Movie) REQUIRE m.title IS UNIQUE")
                except Exception:
                    pass  # If older Neo4j syntax or already exists
                
                # Run the full creation script
                # Neo4j allows executing multi-CREATE statements in a single run
                clean_script = "\n".join([line for line in cql_content.splitlines() if not line.strip().startswith("//")])
                session.run(clean_script)

            return True, "Successfully seeded Neo4j with the Movies dataset!"
        except Exception as e:
            return False, f"Error seeding Neo4j: {str(e)}"

    def execute_cypher(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute Cypher query. If connected to Neo4j, executes remotely.
        Otherwise evaluates using the embedded graph engine.
        """
        params = params or {}
        if self.is_connected and self.driver:
            try:
                with self.driver.session(database=self.database) as session:
                    result = session.run(cypher, params)
                    return [record.data() for record in result]
            except Exception as e:
                # Log error and fallback
                print(f"[Neo4j Error] {e}. Falling back to embedded evaluation.")
                return self.fallback_engine.query_freeform(cypher)
        else:
            return self.fallback_engine.query_freeform(cypher)

    def execute_template(self, template_key: str, params: Dict[str, Any], cypher_query: str) -> List[Dict[str, Any]]:
        """
        Execute a known template query either via live Cypher or embedded graph engine.
        """
        if self.is_connected and self.driver:
            return self.execute_cypher(cypher_query, params)
        else:
            return self.fallback_engine.execute_template(template_key, params)


# Singleton instance helper
_graph_service_instance: Optional[Neo4jGraphService] = None

def get_graph_service(
    uri: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    database: str = "neo4j",
    force_refresh: bool = False
) -> Neo4jGraphService:
    global _graph_service_instance
    if _graph_service_instance is None or force_refresh:
        _graph_service_instance = Neo4jGraphService(uri, user, password, database)
    return _graph_service_instance
