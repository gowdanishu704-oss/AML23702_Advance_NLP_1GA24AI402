"""
Graph RAG (Retrieval-Augmented Generation) Orchestrator.
Coordinates:
1. Natural language question processing and entity recognition
2. Cypher query selection / generation (Template or Dynamic Text-to-Cypher)
3. Graph data retrieval from Neo4j / embedded graph
4. Context formatting and LLM grounded response generation
5. Fallback heuristic response generator when no API key is present
"""

import os
import time
from typing import Dict, Any, List, Optional
from backend.graph_db import get_graph_service, Neo4jGraphService
from backend.query_templates import QUERY_TEMPLATES, get_template
from backend.text2cypher import (
    extract_entities,
    match_template,
    validate_and_sanitize_cypher,
    generate_cypher_prompt
)

try:
    from google import genai
    from google.genai import types
    HAS_GOOGLE_GENAI = True
except ImportError:
    HAS_GOOGLE_GENAI = False


class MovieRAGEngine:
    """
    Graph RAG Engine for Neo4j Movies Knowledge Graph.
    """
    def __init__(
        self,
        graph_service: Optional[Neo4jGraphService] = None,
        gemini_api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash"
    ):
        self.graph_service = graph_service or get_graph_service()
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name
        self._init_llm_client()

    def _init_llm_client(self):
        self.client = None
        if HAS_GOOGLE_GENAI and self.gemini_api_key:
            try:
                self.client = genai.Client(api_key=self.gemini_api_key)
            except Exception as e:
                print(f"[Warning] Failed to initialize Gemini Client: {e}")
                self.client = None

    def set_api_key(self, api_key: str):
        """Update API key at runtime (e.g. from UI input)."""
        self.gemini_api_key = api_key
        self._init_llm_client()

    def _generate_llm_text(self, prompt: str) -> Optional[str]:
        """Call Gemini LLM if configured."""
        if not self.client:
            return None
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=800
                )
            )
            return response.text.strip() if response.text else None
        except Exception as e:
            print(f"[LLM Error] {e}")
            return None

    def _generate_dynamic_cypher(self, question: str) -> Optional[str]:
        """Use LLM to generate dynamic Cypher when template does not match."""
        if not self.client:
            return None
        prompt = generate_cypher_prompt(question)
        raw_cypher = self._generate_llm_text(prompt)
        if not raw_cypher:
            return None
        # Clean formatting backticks
        clean = raw_cypher.replace("```cypher", "").replace("```", "").strip()
        is_safe, sanitized = validate_and_sanitize_cypher(clean)
        if is_safe:
            return sanitized
        return None

    def _build_heuristic_answer(
        self,
        question: str,
        template_id: str,
        records: List[Dict[str, Any]],
        params: Dict[str, Any]
    ) -> str:
        """
        Grounded offline response generator that synthesizes natural language
        answers directly from graph records with zero hallucination.
        """
        if not records:
            return (
                "Based on the Neo4j Movies Knowledge Graph, I could not find any matching "
                f"records for your query. Verified records may be limited to the canonical Movies dataset."
            )

        if template_id == "ACTOR_FILMOGRAPHY":
            actor = params.get("actor", "The actor")
            movies_info = []
            for r in records:
                m_title = r.get("movie")
                rel = r.get("released")
                roles = r.get("roles", [])
                role_str = f" as {', '.join(roles)}" if roles else ""
                movies_info.append(f"• **{m_title}** ({rel}){role_str}")
            return (
                f"According to the Neo4j Knowledge Graph, **{actor}** has starred in **{len(records)}** movie(s):\n\n"
                + "\n".join(movies_info)
                + "\n\n*Retrieved via `:ACTED_IN` graph relationship traversal.*"
            )

        elif template_id == "DIRECTOR_MOVIES":
            director = params.get("director", "The director")
            movies_info = [f"• **{r.get('movie')}** ({r.get('released')}) - *\"{r.get('tagline', '')}\"*" for r in records]
            return (
                f"In the Neo4j Movies dataset, **{director}** is recorded as director of **{len(records)}** film(s):\n\n"
                + "\n".join(movies_info)
                + "\n\n*Retrieved via `:DIRECTED` graph relationship traversal.*"
            )

        elif template_id == "MOVIE_CAST_CREW":
            movie = records[0].get("movie")
            released = records[0].get("released")
            tagline = records[0].get("tagline")
            
            actors = []
            directors = []
            producers = []
            writers = []
            
            for r in records:
                rel = r.get("relationship")
                person = r.get("person")
                roles = r.get("roles", [])
                if rel == "ACTED_IN":
                    role_str = f" (as {', '.join(roles)})" if roles else ""
                    actors.append(f"{person}{role_str}")
                elif rel == "DIRECTED":
                    directors.append(person)
                elif rel == "PRODUCED":
                    producers.append(person)
                elif rel == "WROTE":
                    writers.append(person)
            
            output = [f"### [Movie] **{movie}** ({released})"]
            if tagline:
                output.append(f"> *\"{tagline}\"*\n")
            if directors:
                output.append(f"**Director(s):** {', '.join(directors)}")
            if actors:
                output.append(f"**Cast:** {', '.join(actors)}")
            if writers:
                output.append(f"**Writer(s):** {', '.join(writers)}")
            if producers:
                output.append(f"**Producer(s):** {', '.join(producers)}")
            
            return "\n\n".join(output)

        elif template_id == "CO_ACTORS":
            actor = params.get("actor", "The actor")
            co_actor_lines = []
            for r in records[:6]:
                name = r.get("name")
                count = r.get("count")
                shared = ", ".join(r.get("shared_movies", []))
                co_actor_lines.append(f"- **{name}** -- {count} shared film(s) (*{shared}*)")
            return (
                f"Based on the collaboration subgraph for **{actor}**, frequent co-stars include:\n\n"
                + "\n".join(co_actor_lines)
            )

        elif template_id == "RECOMMEND_BY_ACTOR":
            actor = params.get("actor", "the specified actor")
            rec_lines = []
            for r in records:
                movie = r.get("movie")
                released = r.get("released")
                tagline = r.get("tagline", "")
                via = ", ".join(r.get("via_co_actors", []))
                rec_lines.append(f"- **{movie}** ({released}): Connected via frequent co-stars **{via}**. *\"{tagline}\"*")
            return (
                f"### [Recommendations] Graph-Based Recommendations for fans of **{actor}**\n\n"
                f"These titles are recommended through collaborative graph filtering (2-hop paths connecting co-stars who also starred in these acclaimed titles):\n\n"
                + "\n".join(rec_lines)
            )

        elif template_id == "SHORTEST_PATH":
            r = records[0]
            if "error" in r:
                return f"No path found between {params.get('person1')} and {params.get('person2')} in the movie network."
            path = r.get("path", [])
            path_str = " -> ".join([f"**{node}**" for node in path])
            deg = r.get("degrees_of_separation", len(path)//2)
            return (
                f"### [Connection] Shortest Path Connection\n\n"
                f"Degrees of separation: **{deg}**\n\n"
                f"{path_str}\n\n"
                f"*Traversed via shared movie productions in the knowledge graph.*"
            )

        elif template_id == "SHARED_MOVIES":
            p1 = params.get("person1")
            p2 = params.get("person2")
            movies = [f"- **{r.get('movie')}** ({r.get('released')})" for r in records]
            return (
                f"**{p1}** and **{p2}** have shared the screen in:\n\n"
                + "\n".join(movies)
            )

        elif template_id == "TOP_MOVIES_BY_REVIEWS":
            items = []
            for r in records:
                items.append(f"- **{r.get('movie')}** ({r.get('released')}): Rating **{r.get('avg_rating')}/100** ({r.get('review_count')} reviews)")
            return "### [Top Rated] Highest Rated Movies in Knowledge Graph\n\n" + "\n".join(items)

        # Fallback summary
        return f"Retrieved {len(records)} record(s) from Neo4j:\n\n" + str(records)

    def _synthesize_grounded_answer(
        self,
        question: str,
        records: List[Dict[str, Any]],
        cypher: str
    ) -> str:
        """
        Use LLM with strict grounding prompt to generate natural language answer.
        """
        grounding_prompt = f"""You are CineGraph AI, an expert Movie Knowledge Graph Assistant.
Your task is to answer the user's question using EXCLUSIVELY the provided Neo4j Graph Data.

Rules:
1. Ground every statement strictly in the Graph Data below.
2. DO NOT make up release dates, roles, or connections not in the Graph Data.
3. If the graph data does not contain the answer, explicitly state that it's not present in the current knowledge graph.
4. Format your response cleanly using markdown (bolding, bullet points).
5. If recommendations were returned, explain why they are recommended based on the graph relationships (e.g., co-stars or directors).

User Question: {question}

Executed Cypher Query:
{cypher}

Retrieved Graph Data:
{records}

Grounded Answer:"""

        llm_response = self._generate_llm_text(grounding_prompt)
        return llm_response

    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Full RAG Pipeline Execution:
        1. Query analysis & routing
        2. Cypher execution against Graph
        3. Context formulation
        4. Grounded answer generation
        5. Graph visualization payload construction
        """
        start_time = time.time()
        entities = extract_entities(question)
        match_info = match_template(question)
        
        template_id = None
        params = {}
        cypher_query = ""
        records = []
        routing_mode = "Template Match"

        if match_info:
            template_id = match_info["template_id"]
            params = match_info["params"]
            template_obj = get_template(template_id)
            cypher_query = template_obj["cypher"]
            routing_mode = f"Template: {template_obj['title']}"
            
            # Execute template query
            records = self.graph_service.execute_template(template_id, params, cypher_query)
        else:
            # Fallback to Text-to-Cypher via LLM
            routing_mode = "Text-to-Cypher (Dynamic)"
            generated_cypher = self._generate_dynamic_cypher(question)
            if generated_cypher:
                cypher_query = generated_cypher
                records = self.graph_service.execute_cypher(cypher_query)
            else:
                # Fallback to general scan
                cypher_query = "MATCH (m:Movie) RETURN m.title AS title, m.released AS released LIMIT 10"
                records = self.graph_service.execute_cypher(cypher_query)

        # Generate Grounded Answer (LLM if available, else Heuristic)
        answer = None
        synthesis_source = "Gemini LLM (Grounded)"
        if self.client:
            answer = self._synthesize_grounded_answer(question, records, cypher_query)
        
        if not answer:
            synthesis_source = "Deterministic Graph Engine (Offline Mode)"
            answer = self._build_heuristic_answer(question, template_id or "", records, params)

        elapsed_ms = round((time.time() - start_time) * 1000, 1)

        # Build Graph Visualization Payload
        vis_nodes, vis_edges = self._build_visualization_payload(records, template_id, params)

        return {
            "question": question,
            "answer": answer,
            "cypher_query": cypher_query.strip(),
            "records": records,
            "record_count": len(records),
            "routing_mode": routing_mode,
            "template_id": template_id,
            "params": params,
            "entities": entities,
            "synthesis_source": synthesis_source,
            "execution_time_ms": elapsed_ms,
            "backend_status": self.graph_service.get_status(),
            "graph_visualization": {
                "nodes": vis_nodes,
                "edges": vis_edges
            }
        }

    def _build_visualization_payload(
        self,
        records: List[Dict[str, Any]],
        template_id: Optional[str],
        params: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Transform retrieved records into nodes and edges for graph visualization in UI.
        """
        nodes = {}
        edges = []

        for r in records[:15]:
            # Movie nodes
            m_title = r.get("movie") or r.get("title")
            if m_title and m_title not in nodes:
                nodes[m_title] = {
                    "id": m_title,
                    "label": m_title,
                    "type": "Movie",
                    "year": r.get("released", "")
                }

            # Person nodes
            p_name = r.get("actor") or r.get("director") or r.get("person") or r.get("name")
            if p_name and p_name not in nodes:
                nodes[p_name] = {
                    "id": p_name,
                    "label": p_name,
                    "type": "Person"
                }

            # Relationships
            if p_name and m_title:
                rel_type = r.get("relationship") or ("ACTED_IN" if "roles" in r or "actor" in r else "DIRECTED")
                edges.append({
                    "from": p_name,
                    "to": m_title,
                    "label": rel_type
                })

            # Path nodes
            if "path" in r and isinstance(r["path"], list):
                p_list = r["path"]
                for i in range(len(p_list)):
                    item = p_list[i]
                    if item not in nodes:
                        nodes[item] = {"id": item, "label": item, "type": "PathNode"}
                    if i > 0:
                        edges.append({"from": p_list[i-1], "to": item, "label": "CONNECTED_TO"})

        return list(nodes.values()), edges
