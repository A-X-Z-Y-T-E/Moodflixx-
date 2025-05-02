# Mood-Based Movie Recommendation System

A LangGraph-powered system that analyzes user input to determine mood and preferences, then recommends movies based on those factors using vector search.

## Features

- **Mood Classification**: Analyzes user input to determine emotional state and genre preferences
- **Entertainment Preference Analysis**: Extracts or infers favorite actors, movies, genres, and languages
- **Personalized Reflection Prompts**: Generates thoughtful prompts for users to reflect on their feelings
- **Vector-Based Movie Recommendations**: Uses ChromaDB and embeddings to find relevant movies
- **Movie Mood Analysis**: Determines the mood of movies based on genres and keywords

## Project Structure

- `agent.py`: Main agent workflow using LangGraph for mood analysis and recommendation
- `data_preprocess.py`: Processes and cleans the IMDB 5000 movie dataset
- `create_vector_store.py`: Creates a vector store from processed movie data with mood analysis
- `demo_messages.py`: Sample messages for testing the system

## Setup Instructions

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your API keys:
   ```
   GROQ_API_KEY = "your_groq_api_key"
   ```
4. Download the IMDB 5000 movie dataset and process it:
   ```
   python data_preprocess.py
   ```
5. Create the vector store:
   ```
   python create_vector_store.py
   ```
6. Run the agent:
   ```
   python agent.py
   ```

## How It Works

1. User provides input about their mood, preferences, or movie interests
2. The system analyzes the input to determine mood, genre preferences, and other factors
3. Based on this analysis, it searches the vector store for relevant movies
4. The system returns personalized recommendations with explanations of why each movie matches the user's preferences
5. The user is prompted to reflect on their feelings, creating a more engaging experience

## Technologies Used

- LangGraph for workflow orchestration
- LangChain for LLM integration
- Groq for fast LLM inference
- ChromaDB for vector storage
- HuggingFace Sentence Transformers for embeddings
- Pandas for data processing
