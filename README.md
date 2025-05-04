# 🎬 MoodFlixx - AI Movie Recommendation System

*Discover your next favorite movie through the power of conversation*

![LangChain](https://img.shields.io/badge/LangChain-0.1.0-green)
![LangGraph](https://img.shields.io/badge/LangGraph-0.0.26-blue)
![Llama3](https://img.shields.io/badge/LLM-Llama3--70B-purple)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![TensorFlow Recommenders](https://img.shields.io/badge/TensorFlow--Recommenders-0.7.4-lightblue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-blue)
![FFEM](https://img.shields.io/badge/FFEM-latest-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## ✨ Features

- 🗣 *Conversational AI Recommendations*: Ask for movies in natural language and get personalized suggestions
- 🎭 *Mood-Based Matching*: Our AI understands your mood and recommends movies that fit your current vibe
- 🔍 *Progressive Preference Discovery*: The system asks meaningful questions to better understand your taste
- 📱 *Netflix-Style UI*: Familiar browsing interface with beautiful movie cards
- 🎤 *Multi-Modal Input*: Type queries or use voice input for hands-free interaction
- 📊 *Structured Movie Data*: Returns detailed information including ratings, years, and descriptions

## 💬 Example Conversations

> "I'm feeling sad today, can you recommend something uplifting?"
>
> "Show me sci-fi movies with time travel concepts"
> 
> "I loved Inception and Interstellar - what else might I enjoy?"

## 🚀 Technical Architecture

<table>
<tr>
<td width="50%">

### 🧠 Backend Components

- 🧩 **LangGraph Conversation Flow**: Dynamic preference-gathering conversation with branching paths
- 🔍 **Vector Database**: FAISS for semantic similarity search across thousands of movies
- 📚 **RAG System**: Combines movie database with web search for comprehensive knowledge
- 🔎 **Web Search**: Tavily API integration for supplemental movie information
- 🤖 **LLM Integration**: Powered by Groq's Llama-3 70B model for natural conversations

</td>
<td width="50%">

### 🎨 Frontend Components

- 📱 **Responsive UI**: Adapts beautifully to any device size
- 💬 **Real-Time Chat Widget**: Interactive chatbot with message history
- 🎤 **Voice Input Support**: Speech recognition for hands-free interaction
- 🎞️ **Movie Card Display**: Visual browsing experience with rich metadata
- 🌐 **Progressive Web App**: Works across all modern browsers

</td>
</tr>
</table>

## 📋 API Endpoints

### /chat Endpoint

<details>
<summary>View Request/Response Format (Click to expand)</summary>

*Request format:*
json
{
  "query": "Recommend me some sci-fi movies with time travel",
  "thread_id": "user-123"
}


*Response format:*
json
{
  "response": "Based on your interest in sci-fi time travel movies, here are some recommendations",
  "movies": [
    {
      "title": "Interstellar",
      "imdb_rating": 8.6,
      "year": 2014,
      "description": "A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival.",
      "runtime": 169
    },
    ...
  ],
  "next_question": "What other movie elements are you interested in?"
}

</details>

## 🛠 Setup and Usage

### Detailed Installation Guide

<details>
<summary>🔽 Install with pip (Click to expand)</summary>

#### 1. Clone the Repository
bash
git clone https://github.com/A-X-Z-Y-T-E/Moodflixx-.git
cd Moodflixx-


#### 2. Set Up Python Environment
We recommend using a virtual environment to avoid dependency conflicts:

*Using venv (Python's built-in virtual environment)*:
bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate


#### 3. Install Dependencies
bash
pip install -r requirements.txt


#### 4. Set Up API Keys
Create a .env file in the project root directory with the following content:
bash
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key


#### 5. Run the Application
bash
python agent3.py


</details>

### Using the Chat Interface

- Type your movie preferences or questions in the chat box
- Click the microphone icon for voice input
- Browse movie recommendations in the UI

## 🔍 How It Works

![MoodFlixx Conversation Workflow Graph](workflow_graph.png)

### Conversation Flow Graph Explained

The LangGraph conversation workflow orchestrates the movie recommendation process through several key stages:

1. *Initial Query Analysis* - When a user first interacts with the system, it:
   - Analyzes whether any preferences are already mentioned
   - Either asks for the user's mood or proceeds to recommendation generation

2. *Progressive Preference Collection* - The system follows a carefully designed conversation flow:
   - 🎭 *Mood Collection* - "What mood are you in for a movie today?"
   - 🎬 *Genre Collection* - "What genre would you like to watch?"
   - 📊 *Subgenre Refinement* - "Do you have a specific subgenre preference?"
   - ⏱ *Length Preference* - "Do you prefer short, medium, or long movies?"
   - 📚 *Similar Movies* - "Are there any movies you've enjoyed that you'd like to see something similar to?"
   - 🌟 *Actor Preferences* - "Are there any specific actors you wish to see in your movie?"

3. *Knowledge Retrieval and Processing*:
   - *Semantic or Keyword Search* - The system dynamically determines whether to perform semantic similarity search or keyword-based retrieval
   - *External Information Integration* - Supplements internal movie database with web search results
   - *Content Enhancement* - Combines multiple knowledge sources for comprehensive recommendations

4. *Response Generation* - For each step, the system:
   - Provides relevant movie recommendations based on preferences collected so far
   - Formats results with titles, ratings, and brief descriptions
   - Clearly separates the introduction text, movie list, and next question

This directed graph architecture enables the system to maintain context throughout the conversation while adapting to user inputs and progressively refining recommendations.
# 🎬 AI-Based Movie Recommender System

## 🔧 Designed Pipeline

Our movie recommender system is an *ensemble hybrid model* combining:

- 🔍 *Hybrid-RAG (Retrieval-Augmented Generation)*: Uses both internet search and a curated data corpus.
- 🎯 *Two-Tower Recommendation Engine*: Learns user-item interactions through dual encoders.

We utilize *LLaMA3-70B* as our large language model, embedded within the RAG pipeline.

### 🧠 RAG Details

- *FAISS-based Internet Search Retrieval*
- *Simple Corpus-Based Retrieval*
- Combined to form the context for the *LLM to generate responses*.

### 🧲 Recommendation Engine

- Built on the *Two-Tower Architecture*:
  - *Query Tower*: Encodes the user query (LLM input + metadata + emotion state).
  - *Candidate Tower*: Encodes movies and recommendations.

---

## ⚙ Workflow

1. 🎙 *Speech to Text (STT)*:
   - User provides voice input which is converted to text.

2. 🧾 *Optional Questionnaire*:
   - Tailored to refine preferences (e.g., actors, directors, genres).

3. 🤖 *LLM-RAG Prompt Generation*:
   - Input text sent to the *RAG pipeline* (LLaMA3 + Retrieval).

4. 📸 *Emotion Detection (Bonus Feature)*:
   - Webcam feed processed with *FFEM* and *OpenCV*.
   - Emotional state is extracted and used in query embedding.

5. 🧠 *Two-Tower Query Formation*:
   - Combines:
     - LLM query
     - Emotion
     - Questionnaire answers
     - User metadata (gender, age, region, preferences)

6. 🔁 *Response Generation & Speech*:
   - Generated recommendation is returned and converted to audio using *TTS (Text-to-Speech)*.

---

## 👤 User Metadata Considered

- Gender  
- Age  
- Region  
- Historical preferences  

---

## 📦 Technologies Used

- *LLaMA3-70B*
- *FAISS* (Vector similarity search)
- *FFEM* (Facial emotion analysis)
- *OpenCV* (Video processing)
- *STT & TTS Engines*
- *Two-Tower Recommendation Model*

## 📊 Project Structure

- agent3.py - Main LangGraph application with FastAPI server

- Data files:
  - Vector database: faiss_index/
  - Document store: documents.pkl
  - Movie CSV: data/processed/processed_movies.csv

## 🔮 Future Enhancements

- User accounts and preference history
- Recommendations based on watch history
- Movie trailer integration
- More detailed filtering options
- Direct streaming links

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.