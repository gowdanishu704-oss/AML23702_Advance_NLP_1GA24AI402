"""
Neo4j Seeding Script.
Loads and executes the canonical Neo4j Movies dataset into a live Neo4j instance
(Neo4j Desktop, Neo4j AuraDB, or Docker).
"""

import os
import sys
import argparse
from dotenv import load_dotenv

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.graph_db import Neo4jGraphService


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Seed Neo4j with the Movies dataset.")
    parser.add_argument("--uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687"), help="Neo4j connection URI")
    parser.add_argument("--user", default=os.getenv("NEO4J_USERNAME", "neo4j"), help="Neo4j username")
    parser.add_argument("--password", default=os.getenv("NEO4J_PASSWORD", "password"), help="Neo4j password")
    parser.add_argument("--database", default=os.getenv("NEO4J_DATABASE", "neo4j"), help="Database name")
    parser.add_argument("--cql", default=None, help="Path to custom .cql file")

    args = parser.parse_args()

    print(f"Connecting to Neo4j at {args.uri} (database: {args.database})...")
    service = Neo4jGraphService(
        uri=args.uri,
        user=args.user,
        password=args.password,
        database=args.database
    )

    if not service.is_connected:
        print("[ERROR] Failed to connect to Neo4j.")
        print("Please ensure your Neo4j instance (Neo4j Desktop or AuraDB) is running and credentials are valid.")
        sys.exit(1)

    print("[SUCCESS] Connected to Neo4j instance!")
    print("Executing dataset creation script (data/movies_dataset.cql)...")
    success, msg = service.seed_database(args.cql)

    if success:
        print(f"[COMPLETED] {msg}")
        schema = service.get_schema()
        print("\nGraph Schema Summary:")
        print(f"  • Total Nodes: {schema.get('node_count')}")
        print(f"  • Total Relationships: {schema.get('relationship_count')}")
        print(f"  • Node Labels: {schema.get('labels')}")
        print(f"  • Relationship Types: {schema.get('relationship_types')}")
    else:
        print(f"[ERROR] {msg}")
        sys.exit(1)

    service.close()


if __name__ == "__main__":
    main()
