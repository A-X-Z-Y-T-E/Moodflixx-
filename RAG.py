"""
RAG-based Mood Movie Recommender

This script integrates the mood classification agent with the vector store
to provide personalized movie recommendations based on user's mood and preferences.
"""

import os
from typing import Dict, Any, List
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from agent import process_user_input
import json

# Constants
VECTOR_DB_PATH = "data/vectordb"
TOP_K = 10  # Number of movies to recommend

def load_vector_store(path: str = VECTOR_DB_PATH):
    """
    Load the vector store from the specified path.
    
    Args:
        path: Path to the vector store
        
    Returns:
        Loaded Chroma vector store
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Vector store not found at {path}. Please run create_vector_store.py first.")
    
    # Initialize the embeddings model - must match the one used to create the store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Load the vector store
    vector_store = Chroma(persist_directory=path, embedding_function=embeddings)
    print(f"Loaded vector store from {path}")
    
    return vector_store

def create_search_query(user_features: Dict[str, Any]) -> str:
    """
    Create a search query based on user features extracted by the agent.
    
    Args:
        user_features: Dictionary containing user mood and preferences
        
    Returns:
        Search query string
    """
    # Extract relevant features
    mood = user_features.get("mood_classification", {})
    preferences = user_features.get("entertainment_preferences", {})
    
    # Build query components
    query_components = []
    
    # Add mood-related components
    if mood:
        if "mood" in mood and mood["mood"]:
            query_components.append(f"I'm feeling {mood['mood']}")
        
        if "genre" in mood and mood["genre"]:
            query_components.append(f"I want to watch a {mood['genre']} movie")
        
        if "emotional_tone" in mood and mood["emotional_tone"]:
            query_components.append(f"with a {mood['emotional_tone']} emotional tone")
    
    # Add preference-related components
    if preferences:
        # Add favorite actors if available
        if "favorite_actors" in preferences and preferences["favorite_actors"]:
            actors = preferences["favorite_actors"]
            if isinstance(actors, list) and actors and actors[0] != "unknown":
                actors_str = ", ".join(actors[:3])  # Limit to first 3 actors
                query_components.append(f"starring {actors_str}")
        
        # Add favorite genres if available
        if "genre_preferences" in preferences and preferences["genre_preferences"]:
            genres = preferences["genre_preferences"]
            if isinstance(genres, list) and genres and genres[0] != "unknown":
                genres_str = ", ".join(genres[:3])  # Limit to first 3 genres
                query_components.append(f"in the {genres_str} genre")
    
    # Combine all components into a single query
    query = " ".join(query_components)
    
    # If query is empty, use a default query
    if not query:
        query = "Recommend me a good movie"
    
    print(f"Generated search query: '{query}'")
    return query

def recommend_movies(user_input: str) -> List[Dict[str, Any]]:
    """
    Recommend movies based on user input.
    
    Args:
        user_input: User's message
        
    Returns:
        List of recommended movies with metadata
    """
    # Process user input through the agent to extract features
    print("Processing user input through the agent...")
    user_features = process_user_input(user_input)
    
    # Load the vector store
    vector_store = load_vector_store()
    
    # Create a search query based on user features
    query = create_search_query(user_features)
    
    # Search the vector store
    print(f"Searching for movies matching: '{query}'")
    results = vector_store.similarity_search_with_score(query, k=TOP_K)
    
    # Format the recommendations
    recommendations = []
    for doc, score in results:
        # Extract metadata
        metadata = doc.metadata
        
        # Create a recommendation object
        recommendation = {
            "title": metadata.get("title", "Unknown"),
            "year": metadata.get("year", "Unknown"),
            "director": metadata.get("director", "Unknown"),
            "actors": metadata.get("actors", "Unknown"),
            "genres": metadata.get("genres", "Unknown"),
            "imdb_score": metadata.get("imdb_score", "Unknown"),
            "duration": metadata.get("duration", "Unknown"),
            "language": metadata.get("language", "Unknown"),
            "country": metadata.get("country", "Unknown"),
            "relevance_score": 1 - score,  # Convert distance to similarity score
            "summary": doc.page_content
        }
        
        recommendations.append(recommendation)
    
    return recommendations

def format_recommendations(recommendations: List[Dict[str, Any]]) -> str:
    """
    Format recommendations as a readable string.
    
    Args:
        recommendations: List of movie recommendations
        
    Returns:
        Formatted string with recommendations
    """
    if not recommendations:
        return "Sorry, no movies found matching your mood and preferences."
    
    output = "🎬 Top Movie Recommendations Based on Your Mood:\n\n"
    
    for i, movie in enumerate(recommendations, 1):
        # Format the movie information
        output += f"#{i}: {movie['title']} ({movie['year']})\n"
        output += f"   Director: {movie['director']}\n"
        output += f"   Starring: {movie['actors']}\n"
        output += f"   Genres: {movie['genres']}\n"
        output += f"   IMDB Score: {movie['imdb_score']}/10\n"
        output += f"   Match Score: {movie['relevance_score']:.2f}\n"
        output += f"   Summary: {movie['summary'].strip()}\n\n"
    
    return output

def main():
    """Main function to run the movie recommender."""
    print("=" * 50)
    print("🎭 Mood-Based Movie Recommender 🎬")
    print("=" * 50)
    print("Tell me how you're feeling or what kind of movie you'd like to watch.")
    print("Example: 'I'm feeling sad and need something uplifting' or 'I want an action movie with lots of suspense'")
    print("=" * 50)
    
    while True:
        # Get user input
        user_input = input("\nYour mood or preference: ")
        
        if user_input.lower() in ["exit", "quit", "q"]:
            print("Thank you for using the Mood-Based Movie Recommender. Goodbye!")
            break
        
        try:
            # Get recommendations
            recommendations = recommend_movies(user_input)
            
            # Format and display recommendations
            formatted_recommendations = format_recommendations(recommendations)
            print("\n" + formatted_recommendations)
            
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            print("Please try again with a different query.")

if __name__ == "__main__":
    main()