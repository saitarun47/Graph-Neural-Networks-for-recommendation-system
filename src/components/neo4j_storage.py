import gc
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

def store_data_in_neo4j(data):
    """Store processed data in Neo4j database."""
    print("🚀 Starting optimized Neo4j storage...")
    
    load_dotenv()  
    pwd = os.getenv("NEO4J_PASSWORD")  
    pwd = "gnnproject123"

    driver = GraphDatabase.driver("bolt://neo4j:7687", auth=("neo4j", pwd))
    
    try:
        with driver.session() as session:
            # Clear existing data
            print("🗑️ Clearing existing data...")
            session.run("MATCH (n) DETACH DELETE n")
            
            # ✅ BATCH CREATE USERS (much more efficient)
            print(f"👥 Creating {len(data['user_features'])} users in batch...")
            user_batch = []
            for i, features in enumerate(data['user_features']):
                user_batch.append({
                    'user_id': i, 
                    'age': float(features[0]), 
                    'occ': float(features[1]),
                    'avg_rat': float(features[2]), 
                    'num_rat': float(features[3])
                })
                
                # Process in chunks of 1000 to manage memory
                if len(user_batch) >= 1000 or i == len(data['user_features']) - 1:
                    session.run("""
                        UNWIND $users as user
                        CREATE (u:User {
                            id: user.user_id,
                            age_scaled: user.age,
                            occupation: user.occ,
                            avg_ratings: user.avg_rat,
                            num_ratings: user.num_rat
                        })
                    """, users=user_batch)
                    print(f"✅ Created {len(user_batch)} users")
                    user_batch = []  # Clear batch
                    gc.collect()  # Force garbage collection
            
            # ✅ BATCH CREATE MOVIES
            print(f"🎬 Creating {len(data['movie_features'])} movies in batch...")
            movie_batch = []
            for i, features in enumerate(data['movie_features']):
                movie_batch.append({
                    'movie_id': i,
                    'genres': [float(x) for x in features[:-2]],
                    'avg_rating': float(features[-2]),
                    'num_ratings': float(features[-1])
                })
                
                # Process in chunks
                if len(movie_batch) >= 1000 or i == len(data['movie_features']) - 1:
                    session.run("""
                        UNWIND $movies as movie
                        CREATE (m:Movie {
                            id: movie.movie_id,
                            genres: movie.genres,
                            avg_rating: movie.avg_rating,
                            num_ratings: movie.num_ratings
                        })
                    """, movies=movie_batch)
                    print(f"✅ Created {len(movie_batch)} movies")
                    movie_batch = []
                    gc.collect()
            
            # ✅ BATCH CREATE RELATIONSHIPS
            splits = data['splits']
            
            def create_edges_batch(edges, split_name, label):
                print(f"🔗 Creating {len(edges)} {split_name} edges (label={label})...")
                edge_batch = []
                
                for edge in edges:
                    edge_batch.append({
                        'user_id': edge['user_id_mapped'],
                        'movie_id': edge['movie_id_mapped'],
                        'split': split_name,
                        'label': label
                    })
                    
                    # Process in chunks of 5000
                    if len(edge_batch) >= 5000:
                        session.run("""
                            UNWIND $edges as edge
                            MATCH (u:User {id: edge.user_id})
                            MATCH (m:Movie {id: edge.movie_id})
                            CREATE (u)-[:RATES {split: edge.split, label: edge.label}]->(m)
                        """, edges=edge_batch)
                        print(f"✅ Created {len(edge_batch)} edges")
                        edge_batch = []
                        gc.collect()
                
                # Process remaining edges
                if edge_batch:
                    session.run("""
                        UNWIND $edges as edge
                        MATCH (u:User {id: edge.user_id})
                        MATCH (m:Movie {id: edge.movie_id})
                        CREATE (u)-[:RATES {split: edge.split, label: edge.label}]->(m)
                    """, edges=edge_batch)
                    print(f"✅ Created final {len(edge_batch)} edges")

            # Create all relationship types
            create_edges_batch(splits['train_pos'], 'train', 1)
            create_edges_batch(splits['train_neg'], 'train', 0)
            create_edges_batch(splits['val_pos'], 'val', 1)
            create_edges_batch(splits['val_neg'], 'val', 0)
            create_edges_batch(splits['test_pos'], 'test', 1)
            create_edges_batch(splits['test_neg'], 'test', 0)
            
    finally:
        driver.close()
        gc.collect()  # Final cleanup
    
    print("🎉 Neo4j storage completed successfully!")
    
    return {
        "message": "Data stored in Neo4j successfully",
        "stats": {
            "users": len(data['user_features']),
            "movies": len(data['movie_features']),
            "train_edges": len(splits['train_pos']) + len(splits['train_neg']),
            "val_edges": len(splits['val_pos']) + len(splits['val_neg']),
            "test_edges": len(splits['test_pos']) + len(splits['test_neg'])
        }
    }
