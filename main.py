from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import torch
import logging
import time
import sys
import os
from functools import wraps


from prometheus_fastapi_instrumentator import Instrumentator
import statsd


sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.components.model import SimpleGraphSAGE
from src.components.data_loader import load_raw_data
from src.components.graph_builder import build_heterogeneous_graph

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="GNN Movie Recommendation API",
    description="A GraphSAGE-based movie recommendation system with monitoring",
    version="1.0.0"
)

# Prometheus instrumentation
instrumentator = Instrumentator(
    should_group_status_codes=False,
    should_ignore_untemplated=True,
    should_respect_env_var=True,
    should_instrument_requests_inprogress=True,
    excluded_handlers=[".*admin.*"],
    env_var_name="ENABLE_METRICS",
    inprogress_name="inprogress",
    inprogress_labels=True,
)

instrumentator.instrument(app).expose(app)

# Initialize StatsD client for custom metrics
statsd_client = None
try:
    statsd_host = os.getenv('STATSD_HOST', 'localhost')
    statsd_port = int(os.getenv('STATSD_PORT', 9125))
    statsd_client = statsd.StatsClient(statsd_host, statsd_port, prefix='graphsage')
    logger.info(f"StatsD client initialized: {statsd_host}:{statsd_port}")
except Exception as e:
    logger.warning(f"Failed to initialize StatsD client: {e}")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom metrics decorator
def track_recommendations(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = await func(*args, **kwargs)
            
            # Track metrics
            if statsd_client:
                duration = time.time() - start_time
                user_type = result.user_type if hasattr(result, 'user_type') else 'unknown'
                strategy = result.strategy_used if hasattr(result, 'strategy_used') else 'unknown'
                
                statsd_client.timing(f'recommendations.duration_seconds.{user_type}.{strategy}', duration * 1000)
                statsd_client.incr(f'recommendations.total.{user_type}.{strategy}')
                
                # Track prediction scores for existing users
                if hasattr(result, 'recommendations') and result.recommendations:
                    avg_score = sum(r.get('prediction_score', 0) for r in result.recommendations) / len(result.recommendations)
                    statsd_client.gauge(f'model.prediction_score.{user_type}', avg_score)
            
            return result
            
        except Exception as e:
            if statsd_client:
                statsd_client.incr('recommendations.errors')
            raise
    
    return wrapper

# Add metrics middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    # Track API response times
    if statsd_client:
        endpoint = request.url.path
        method = request.method
        status_code = response.status_code
        statsd_client.timing(f'api.request_duration.{method}.{endpoint}', process_time * 1000)
        statsd_client.incr(f'api.requests.{method}.{status_code}')
    
    return response

# Global variables
model = None
graph_data = None
user_id_map = None
movie_id_map = None
reverse_user_map = None
reverse_movie_map = None
device = None

# Pydantic models
class RecommendationRequest(BaseModel):
    user_id: Optional[int] = None
    user_profile: Optional[Dict[str, Any]] = None
    num_recommendations: Optional[int] = 10
    strategy: Optional[str] = "popular"  

class RecommendationResponse(BaseModel):
    user_id: Optional[int]
    recommendations: List[Dict[str, Any]]
    strategy_used: str
    user_type: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    total_users: int
    total_movies: int

def load_model_and_data():
    """Load the trained model and data"""
    global model, graph_data, user_id_map, movie_id_map, reverse_user_map, reverse_movie_map, device
    
    try:
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Load data
        raw_data = load_raw_data()
        graph_data = build_heterogeneous_graph(raw_data)
        
        # Create mappings
        user_id_map = {raw: idx for idx, raw in enumerate(raw_data['users']['user_id'])}
        movie_id_map = {raw: idx for idx, raw in enumerate(raw_data['items']['movie_id'])}
        reverse_user_map = {idx: raw for raw, idx in user_id_map.items()}
        reverse_movie_map = {idx: raw for raw, idx in movie_id_map.items()}
        
        # Load model
        model = SimpleGraphSAGE(
            in_feats=graph_data['x'].size(1), 
            hidden_dim=128
        ).to(device)
        
        # Load trained weights
        model.load_state_dict(torch.load('models/best_graphsage_model.pth', map_location=device))
        model.eval()
        
        logger.info("Model and data loaded successfully")
        
        # Track model loading metrics
        if statsd_client:
            statsd_client.incr('model.loaded_successfully')
            statsd_client.gauge('model.total_users', len(user_id_map))
            statsd_client.gauge('model.total_movies', len(movie_id_map))
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to load model and data: {e}")
        
        # Track model loading errors
        if statsd_client:
            statsd_client.incr('model.load_errors')
        
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    logger.info("Starting GNN Recommendation API...")
    success = load_model_and_data()
    if not success:
        logger.error("Failed to initialize application")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        total_users=len(user_id_map) if user_id_map else 0,
        total_movies=len(movie_id_map) if movie_id_map else 0
    )


@app.post("/recommendations", response_model=RecommendationResponse)
@track_recommendations
async def get_recommendations(request: RecommendationRequest):
    """Get movie recommendations"""
    start_time = time.time()
    
    try:
        if request.user_id is not None:
            # Existing user
            if request.user_id not in user_id_map:
                raise HTTPException(status_code=404, detail=f"User {request.user_id} not found")
            
            recommendations = await get_existing_user_recommendations(
                request.user_id, 
                request.num_recommendations
            )
            user_type = "existing"
            strategy_used = "gnn_model"
            
        else:
            # New user
            if not request.user_profile:
                raise HTTPException(status_code=400, detail="User profile required for new users")
            
            recommendations = get_new_user_recommendations(
                request.user_profile,
                request.num_recommendations,
                request.strategy
            )
            user_type = "new"
            strategy_used = request.strategy
        
        processing_time = time.time() - start_time
        logger.info(f"Generated {len(recommendations)} recommendations in {processing_time:.2f}s")
        
        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=recommendations,
            strategy_used=strategy_used,
            user_type=user_type
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

async def get_existing_user_recommendations(user_id: int, num_recommendations: int) -> List[Dict[str, Any]]:
    """Get recommendations for existing users using GNN model"""
    
    user_mapped_id = user_id_map[user_id]
    
    # Get all movies
    all_movies = list(movie_id_map.values())
    
    # Create edge indices for prediction
    edge_indices = []
    for movie_id in all_movies:
        edge_indices.append([user_mapped_id, movie_id])
    
    edge_indices = torch.tensor(edge_indices, dtype=torch.long).t().to(device)
    
    # Get predictions
    with torch.no_grad():
        predictions = model(
            graph_data['x'].to(device),
            graph_data['edge_index'].to(device),
            edge_indices
        )
        scores = torch.sigmoid(predictions).cpu().numpy()
    
    # Sort by prediction scores
    movie_scores = list(zip(all_movies, scores))
    movie_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Get top recommendations
    recommendations = []
    for movie_mapped_id, score in movie_scores[:num_recommendations]:
        movie_raw_id = reverse_movie_map[movie_mapped_id]
        
        # Get movie info
        movie_idx = None
        for i, mid in enumerate(graph_data['items']['movie_id']):
            if mid == movie_raw_id:
                movie_idx = i
                break
        
        if movie_idx is not None:
            movie_info = {
                'movie_id': movie_raw_id,
                'title': graph_data['items']['movie_title'][movie_idx],
                'genres': extract_genres(graph_data['items'], movie_idx),
                'prediction_score': float(score),
                'rank': len(recommendations) + 1
            }
            recommendations.append(movie_info)
    
    return recommendations

def get_new_user_recommendations(user_profile: Dict[str, Any], 
                               num_recommendations: int,
                               strategy: str) -> List[Dict[str, Any]]:
    """Get recommendations for new users"""
    
    if strategy == "popular":
        return get_popular_recommendations(num_recommendations)
    elif strategy == "content_based":
        return get_content_based_recommendations(user_profile, num_recommendations)
    elif strategy == "hybrid":
        return get_hybrid_recommendations(user_profile, num_recommendations)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown strategy: {strategy}")

def get_popular_recommendations(num_recommendations: int) -> List[Dict[str, Any]]:
    """Get popular movie recommendations"""
    # Calculate movie popularity 
    movie_ratings = {}
    for i, movie_id in enumerate(graph_data['ratings']['movie_id_mapped']):
        rating = graph_data['ratings']['rating'][i]
        if movie_id not in movie_ratings:
            movie_ratings[movie_id] = {'total': 0, 'count': 0}
        movie_ratings[movie_id]['total'] += rating
        movie_ratings[movie_id]['count'] += 1
    
    # Calculate average ratings
    movie_scores = []
    for movie_id, stats in movie_ratings.items():
        avg_rating = stats['total'] / stats['count']
        popularity = stats['count'] * avg_rating
        movie_scores.append((movie_id, popularity))
    
    # Sort by popularity
    movie_scores.sort(key=lambda x: x[1], reverse=True)
    
    recommendations = []
    for movie_mapped_id, popularity_score in movie_scores[:num_recommendations]:
        movie_raw_id = reverse_movie_map[movie_mapped_id]
        
        # Get movie info
        movie_idx = None
        for i, mid in enumerate(graph_data['items']['movie_id']):
            if mid == movie_raw_id:
                movie_idx = i
                break
        
        if movie_idx is not None:
            movie_info = {
                'movie_id': movie_raw_id,
                'title': graph_data['items']['movie_title'][movie_idx],
                'genres': extract_genres(graph_data['items'], movie_idx),
                'popularity_score': float(popularity_score),
                'rank': len(recommendations) + 1
            }
            recommendations.append(movie_info)
    
    return recommendations

def get_content_based_recommendations(user_profile: Dict[str, Any], 
                                    num_recommendations: int) -> List[Dict[str, Any]]:
    """Get content-based recommendations"""
    
    preferred_genres = user_profile.get('preferred_genres', [])
    age = user_profile.get('age', 25)
    
    # Simple genre matching
    genre_columns = [
        "Action", "Adventure", "Animation", "Children's", "Comedy",
        "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", 
        "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", 
        "Thriller", "War", "Western"
    ]
    
    # Score movies based on genre preferences
    movie_scores = []
    for i, movie_id in enumerate(graph_data['items']['movie_id']):
        score = 0
        for genre in preferred_genres:
            if genre in genre_columns:
                genre_idx = genre_columns.index(genre)
                if genre_idx < len(graph_data['items']) and graph_data['items'][genre][i] == 1:
                    score += 1
        
        movie_scores.append((movie_id, score))
    
    # Sort by score
    movie_scores.sort(key=lambda x: x[1], reverse=True)
    
    recommendations = []
    for movie_raw_id, score in movie_scores[:num_recommendations]:
        if score > 0:  # Only include movies that match preferences
            movie_idx = None
            for i, mid in enumerate(graph_data['items']['movie_id']):
                if mid == movie_raw_id:
                    movie_idx = i
                    break
            
            if movie_idx is not None:
                movie_info = {
                    'movie_id': movie_raw_id,
                    'title': graph_data['items']['movie_title'][movie_idx],
                    'genres': extract_genres(graph_data['items'], movie_idx),
                    'genre_match_score': score,
                    'rank': len(recommendations) + 1
                }
                recommendations.append(movie_info)
    
    return recommendations

def get_hybrid_recommendations(user_profile: Dict[str, Any], 
                             num_recommendations: int) -> List[Dict[str, Any]]:
    """Get hybrid recommendations"""
    
    # Get recommendations from both strategies
    popular_recs = get_popular_recommendations(num_recommendations * 2)
    content_recs = get_content_based_recommendations(user_profile, num_recommendations * 2)
    
    # Combine and deduplicate
    seen_movies = set()
    recommendations = []
    
    # Add content-based first (more personalized)
    for rec in content_recs:
        if rec['movie_id'] not in seen_movies and len(recommendations) < num_recommendations:
            rec['strategy'] = 'content_based'
            recommendations.append(rec)
            seen_movies.add(rec['movie_id'])
    
    # Fill remaining with popular movies
    for rec in popular_recs:
        if rec['movie_id'] not in seen_movies and len(recommendations) < num_recommendations:
            rec['strategy'] = 'popular'
            recommendations.append(rec)
            seen_movies.add(rec['movie_id'])
    
    return recommendations

def extract_genres(items_data: Dict, movie_idx: int) -> List[str]:
    """Extract genres for a movie"""
    genre_columns = [
        "Action", "Adventure", "Animation", "Children's", "Comedy",
        "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", 
        "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", 
        "Thriller", "War", "Western"
    ]
    
    genres = []
    for genre in genre_columns:
        if genre in items_data and items_data[genre][movie_idx] == 1:
            genres.append(genre)
    
    return genres

@app.get("/users/{user_id}")
async def get_user_info(user_id: int):
    """Get user information"""
    try:
        if user_id not in user_id_map:
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")
        
        # Find user in data
        user_idx = None
        for i, uid in enumerate(graph_data['users']['user_id']):
            if uid == user_id:
                user_idx = i
                break
        
        if user_idx is None:
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")
        
        # Count user's ratings
        user_mapped_id = user_id_map[user_id]
        rating_count = sum(1 for uid in graph_data['ratings']['user_id_mapped'] if uid == user_mapped_id)
        
        return {
            "user_id": user_id,
            "age": graph_data['users']['age'][user_idx] if 'age' in graph_data['users'] else None,
            "gender": graph_data['users']['gender'][user_idx] if 'gender' in graph_data['users'] else None,
            "occupation": graph_data['users']['occupation'][user_idx] if 'occupation' in graph_data['users'] else None,
            "total_ratings": rating_count
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user info: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/movies/{movie_id}")
async def get_movie_info(movie_id: int):
    """Get movie information"""
    try:
        if movie_id not in movie_id_map:
            raise HTTPException(status_code=404, detail=f"Movie {movie_id} not found")
        
        # Find movie in data
        movie_idx = None
        for i, mid in enumerate(graph_data['items']['movie_id']):
            if mid == movie_id:
                movie_idx = i
                break
        
        if movie_idx is None:
            raise HTTPException(status_code=404, detail=f"Movie {movie_id} not found")
        
        return {
            "movie_id": movie_id,
            "title": graph_data['items']['movie_title'][movie_idx],
            "genres": extract_genres(graph_data['items'], movie_idx),
            "release_date": graph_data['items']['release_date'][movie_idx] if 'release_date' in graph_data['items'] else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting movie info: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    try:
        return {
            "total_users": len(user_id_map) if user_id_map else 0,
            "total_movies": len(movie_id_map) if movie_id_map else 0,
            "total_ratings": len(graph_data['ratings']['user_id_mapped']) if graph_data else 0,
            "model_loaded": model is not None,
            "device": str(device) if device else "unknown"
        }
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    
@app.post("/webhook/retrain")
async def trigger_retraining(request: Request):
    """Webhook endpoint to trigger model retraining"""
    try:
        # Log the alert
        logger.warning("Retraining webhook triggered by alert")
        
        # Track the event
        if statsd_client:
            statsd_client.incr('model.retraining_triggered')
        
        # Trigger Airflow DAG
        import requests
        airflow_url = "http://airflow-webserver:8080/api/v1/dags/retrain_graphsage/dagRuns"
        airflow_auth = ("airflow", "airflow")
        
        payload = {
            "conf": {
                "triggered_by": "performance_alert",
                "timestamp": time.time()
            }
        }
        
        response = requests.post(airflow_url, json=payload, auth=airflow_auth)
        
        if response.status_code == 200:
            logger.info(" Retraining pipeline triggered successfully")
            return {"status": "success", "message": "Retraining pipeline triggered"}
        else:
            logger.error(f"Failed to trigger retraining: {response.text}")
            return {"status": "error", "message": "Failed to trigger retraining"}
            
    except Exception as e:
        logger.error(f"Error in retraining webhook: {e}")
        raise HTTPException(status_code=500, detail="Webhook processing failed")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
