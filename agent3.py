from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END, add_messages
from langgraph.checkpoint.memory import MemorySaver
import pandas as pd
import os
import json
import ast
import pickle
import torch
from dotenv import load_dotenv
from pydantic import BaseModel as PydanticBaseModel 
from typing import List, Optional, Dict, Any, TypedDict, Annotated

from langchain_community.vectorstores import FAISS
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from fastapi import FastAPI 

load_dotenv()

class MovieRecState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    context: Optional[List[Document]]
    query: Optional[str]
    response: Optional[str]
    retrieval_type: str 
    web_results: Optional[List[Dict[str, Any]]]
    collected_mood: Optional[str]
    collected_genre: Optional[str]
    collected_subgenre: Optional[str]
    collected_length: Optional[str]
    collected_directors: Optional[str]  # Added this field for directors
    collected_actors: Optional[str]     # Added this field for actors

class ChatRequest(PydanticBaseModel):
    query: str
    thread_id: str 

class ChatResponse(PydanticBaseModel):
    response: str

device = "cuda" if torch.cuda.is_available() else "cpu"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True},
)

def create_vector_db(csv_path, faiss_index_path, pickle_path):
    if os.path.exists(faiss_index_path) and os.path.exists(pickle_path):
        try:
            with open(pickle_path, "rb") as f:
                stored_docs = pickle.load(f)
            vectordb = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
            return vectordb
        except Exception as e:
            print(f"Error loading existing vector DB: {e}")

    df = pd.read_csv(csv_path)
    documents = []
    for idx, row in df.iterrows():
        try:
            # --- Robust Genre Parsing --- 
            genres_raw = row.get('genres') # Use .get() for safety
            genres_list = []
            if pd.isna(genres_raw):
                genres = "N/A"
            elif isinstance(genres_raw, str):
                try:
                    # Attempt safe evaluation first
                    evaluated = ast.literal_eval(genres_raw)
                    if isinstance(evaluated, list):
                        genres_list = [str(g).strip() for g in evaluated] # Ensure items are strings
                    elif isinstance(evaluated, str): # Handle case like '"Action"'
                         genres_list = [evaluated.strip()]
                    else:
                         genres_list = [str(evaluated).strip()] # Handle other literals
                except (ValueError, SyntaxError):
                    # Fallback: Treat as comma-separated string
                    genres_list = [g.strip() for g in genres_raw.split(',') if g.strip()]
                genres = ", ".join(genres_list) if genres_list else "N/A"
            elif isinstance(genres_raw, list): # Handle if it's already a list somehow
                 genres_list = [str(g).strip() for g in genres_raw]
                 genres = ", ".join(genres_list) if genres_list else "N/A"
            else:
                # Handle other unexpected types
                genres = "N/A"
            # --- End Robust Genre Parsing ---
            
            # --- Safely Access Row Data ---
            movie_id = row.get('id', f'MISSING_ID_{idx}') # Use index as fallback ID
            title = row.get('title', 'Unknown Title')
            year = row.get('year', 'Unknown Year')
            rating = row.get('rating', 'N/A')
            plot_summary = row.get('plot_summary', 'No summary available.')
            # --- End Safe Access ---
            
            metadata = {
                "source": f"{movie_id}-{title}",
                "title": title,
                "year": year,
                "genres": genres, # Already handled above
                "rating": rating
            }
            page_content = f"Title: {title}\nYear: {year}\nGenres: {genres}\nRating: {rating}\nPlot Summary: {plot_summary}"
            documents.append(Document(page_content=page_content, metadata=metadata))
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
    
    if not documents:
        print("No documents were created from the CSV.")
        return None

    vectordb = FAISS.from_documents(documents, embeddings)
    vectordb.save_local(faiss_index_path)
    with open(pickle_path, "wb") as f:
        pickle.dump(documents, f)
    return vectordb

CSV_PATH = "data\processed\processed_movies.csv"
FAISS_INDEX_PATH = "faiss_index"
PICKLE_PATH = "documents.pkl"
vector_db = create_vector_db(CSV_PATH, FAISS_INDEX_PATH, PICKLE_PATH)

def get_retriever(k_value=5):
    if vector_db is None:
        raise ValueError("Vector DB is not initialized. Cannot create retriever.")
    return vector_db.as_retriever(search_kwargs={"k": k_value})

groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="llama3-70b-8192")
tavily_tool = TavilySearchResults(max_results=3)

QUERY_ANALYZER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert query analyzer. Your task is to determine the type of retrieval required for a user's movie query."
            "Analyze the user's message and classify it into one of the following categories based ONLY on the user's explicit request:"
            "'semantic' - For queries asking for movies based on plot similarity, themes, mood, or general descriptions (e.g., 'movies like Inception', 'uplifting films')."
            "'keyword' - For queries asking for movies based on specific factual criteria like actors, directors, genres, titles, or years (e.g., 'movies starring Tom Hanks', 'sci-fi movies from the 90s', 'films directed by Nolan')."
            "'none' - For greetings, conversational fillers, or questions not related to movie recommendations (e.g., 'hello', 'how are you?', 'tell me a joke')."
            "Respond ONLY with 'semantic', 'keyword', or 'none'."
        ),
        ("human", "{question}"),
    ]
)

analyzer_chain = QUERY_ANALYZER_PROMPT | llm | StrOutputParser()

def analyze_query(state: MovieRecState) -> Dict:
    if not state["messages"]:
        return {"retrieval_type": "none", "query": None}
    
    last_message = state["messages"][-1]
    if not isinstance(last_message, HumanMessage):
        return {"retrieval_type": "none", "query": None}
        
    user_question = last_message.content
    retrieval_type = analyzer_chain.invoke({"question": user_question})
    return {"retrieval_type": retrieval_type.strip().lower(), "query": user_question}

def adaptive_retrieval(state: MovieRecState) -> Dict:
    query = state.get("query")
    current_retrieval_type = state.get("retrieval_type", "none") 
    retrieved_docs = []

    if query and current_retrieval_type != "none":
        try:
            retriever = get_retriever() 
            retrieved_docs = retriever.invoke(query)
        except Exception as e:
            print(f"Error during FAISS retrieval: {e}")
            
    return {"context": retrieved_docs}

def web_search(state: MovieRecState) -> Dict:
    query = state.get("query")
    web_results = []
    if query:
        try:
            web_results = tavily_tool.invoke({"query": query})
        except Exception as e:
            print(f"Error during Tavily web search: {e}")
    
    return {"web_results": web_results}

def parse_user_preferences(state: MovieRecState) -> Dict:
    """
    Parse user messages to extract and update preferences from any message,
    not just in response to specific questions.
    """
    messages = state['messages']
    
    # Get current state values with defaults
    collected_mood = state.get("collected_mood")
    collected_genre = state.get("collected_genre") 
    collected_subgenre = state.get("collected_subgenre")
    collected_length = state.get("collected_length")
    collected_directors = state.get("collected_directors")
    collected_actors = state.get("collected_actors")
    
    # Only process if we have at least one message with user input
    if not messages or not isinstance(messages[-1], HumanMessage):
        return {}
    
    # Get the latest user message and our previous message if exists
    latest_user_msg = messages[-1].content
    prev_ai_msg = ""
    
    # Find the most recent AI message if it exists
    for msg in reversed(messages[:-1]):
        if isinstance(msg, AIMessage):
            prev_ai_msg = msg.content.lower()
            break
    
    updates = {}
    
    # Enhanced batch preference parser - looks for all preferences at once
    batch_parse_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a preference parser specialized in movie recommendations.
        Analyze the user's message and extract ALL preferences mentioned for watching movies.
        Format your response as a valid JSON object with ONLY these keys: "mood", "genre", "subgenre", "length", "directors", "actors".
        Use null for any preference not found in the message.
        Example response: {"mood": "relaxing", "genre": "comedy", "subgenre": null, "length": "short", "directors": null, "actors": null}
        """),
        ("human", """User message: {user_msg}
        Previous assistant message (if any): {prev_ai_msg}
        
        Extract ALL movie preferences from this message. Return ONLY the JSON object.
        """)
    ])
    
    # If this is the first turn or if the message seems to contain multiple preferences,
    # use the batch parser to try to extract everything at once
    if len(messages) <= 2 or len(latest_user_msg.split()) > 15:
        try:
            batch_parse_chain = batch_parse_prompt | llm | StrOutputParser()
            json_response = batch_parse_chain.invoke({
                "user_msg": latest_user_msg,
                "prev_ai_msg": prev_ai_msg
            })
            
            try:
                parsed_prefs = json.loads(json_response)
                print(f"Batch parsed preferences: {parsed_prefs}")
                
                # Only update preferences that aren't already set
                if not collected_mood and parsed_prefs.get("mood"):
                    updates["collected_mood"] = parsed_prefs["mood"]
                    print(f"Updated mood from batch parser: {parsed_prefs['mood']}")
                    
                if not collected_genre and parsed_prefs.get("genre"):
                    updates["collected_genre"] = parsed_prefs["genre"]
                    print(f"Updated genre from batch parser: {parsed_prefs['genre']}")
                    
                if not collected_subgenre and parsed_prefs.get("subgenre"):
                    updates["collected_subgenre"] = parsed_prefs["subgenre"]
                    print(f"Updated subgenre from batch parser: {parsed_prefs['subgenre']}")
                    
                if not collected_length and parsed_prefs.get("length"):
                    updates["collected_length"] = parsed_prefs["length"]
                    print(f"Updated length from batch parser: {parsed_prefs['length']}")
                    
                if not collected_directors and parsed_prefs.get("directors"):
                    updates["collected_directors"] = parsed_prefs["directors"]
                    print(f"Updated directors from batch parser: {parsed_prefs['directors']}")
                    
                if not collected_actors and parsed_prefs.get("actors"):
                    updates["collected_actors"] = parsed_prefs["actors"]
                    print(f"Updated actors from batch parser: {parsed_prefs['actors']}")
                
            except json.JSONDecodeError:
                print("Failed to parse JSON response from batch parser")
                
        except Exception as e:
            print(f"Error in batch preference parsing: {e}")
    
    # Use the targeted parser as a fallback for specific preference questions
    if not updates:
        parse_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a preference parser. Extract user preferences from their message based on context.
            Respond ONLY with the extracted value or "null" if not found.
            Do not include any explanation or additional text."""),
            ("human", """Previous AI Message: {prev_ai_msg}
            User Message: {user_msg}
            Looking for: {looking_for}
            
            Extract ONLY the user's {looking_for} preference. Respond with just that value or "null".
            """)
        ])
        
        parse_chain = parse_prompt | llm | StrOutputParser()
        
        # Based on current state, determine what to parse
        try:
            # First collect mood if not present
            if not collected_mood:
                if "mood" in prev_ai_msg or len(messages) <= 2:
                    mood = parse_chain.invoke({
                        "prev_ai_msg": prev_ai_msg,
                        "user_msg": latest_user_msg,
                        "looking_for": "mood"
                    }).strip()
                    
                    if mood.lower() != "null":
                        updates["collected_mood"] = mood
                        print(f"Parsed mood: {mood}")
                        
            # Then collect genre if mood is present but genre isn't
            elif not collected_genre:
                if "genre" in prev_ai_msg or len(messages) <= 2:
                    genre = parse_chain.invoke({
                        "prev_ai_msg": prev_ai_msg,
                        "user_msg": latest_user_msg,
                        "looking_for": "genre"
                    }).strip()
                    
                    if genre.lower() != "null":
                        updates["collected_genre"] = genre
                        print(f"Parsed genre: {genre}")
            
            # Then collect subgenre if mood and genre are present but subgenre isn't            
            elif not collected_subgenre:
                if "subgenre" in prev_ai_msg or len(messages) <= 2:
                    subgenre = parse_chain.invoke({
                        "prev_ai_msg": prev_ai_msg,
                        "user_msg": latest_user_msg,
                        "looking_for": "subgenre"
                    }).strip()
                    
                    if subgenre.lower() != "null":
                        updates["collected_subgenre"] = subgenre
                        print(f"Parsed subgenre: {subgenre}")
                        
            # Collect length if mood, genre, and subgenre are present
            elif not collected_length:
                if "length" in prev_ai_msg or len(messages) <= 2:
                    length = parse_chain.invoke({
                        "prev_ai_msg": prev_ai_msg,
                        "user_msg": latest_user_msg,
                        "looking_for": "length"
                    }).strip()
                    
                    if length.lower() != "null":
                        updates["collected_length"] = length
                        print(f"Parsed length: {length}")
            
            # Collect directors if previous preferences are present but directors aren't
            elif not collected_directors:
                if "director" in prev_ai_msg or len(messages) <= 2:
                    directors = parse_chain.invoke({
                        "prev_ai_msg": prev_ai_msg,
                        "user_msg": latest_user_msg,
                        "looking_for": "directors"
                    }).strip()
                    
                    if directors.lower() != "null":
                        updates["collected_directors"] = directors
                        print(f"Parsed directors: {directors}")
                        
            # Finally collect actors if all other preferences are present
            elif not collected_actors:
                if "actor" in prev_ai_msg or len(messages) <= 2:
                    actors = parse_chain.invoke({
                        "prev_ai_msg": prev_ai_msg,
                        "user_msg": latest_user_msg,
                        "looking_for": "actors"
                    }).strip()
                    
                    if actors.lower() != "null":
                        updates["collected_actors"] = actors
                        print(f"Parsed actors: {actors}")
        except Exception as e:
            print(f"Error parsing user preferences: {e}")
    
    return updates

def analyze_initial_input(state: MovieRecState) -> Dict:
    """
    Process the initial user input to extract any preferences mentioned upfront.
    This allows the system to skip steps when preferences are already specified.
    """
    if not state["messages"] or not isinstance(state["messages"][-1], HumanMessage):
        return {}
    
    # Get preferences from the first user message
    preference_updates = parse_user_preferences(state)
    
    print(f"Initial preference analysis: {preference_updates}")
    
    # Return the detected preferences
    return preference_updates

RESPONSE_GENERATOR_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a conversational movie recommendation assistant that adapts to the user's preferences.
Your goal is to gather MOOD, GENRE, SUBGENRE, LENGTH, DIRECTORS, and ACTORS preferences, but you should skip asking for any preferences that are already collected.

**Dynamic Conversation Flow:**
1. **Analyze State:** Look at what preferences are already collected: `collected_mood`, `collected_genre`, `collected_subgenre`, `collected_length`, `collected_directors`, `collected_actors`.

2. **Determine Next Question:**
    - If `collected_mood` is 'Missing': Ask "What kind of mood are you in for a movie?"
    - Else if `collected_genre` is 'Missing': Ask about genre and recommend based on mood.
    - Else if `collected_subgenre` is 'Missing': Ask about subgenre and recommend based on mood + genre.
    - Else if `collected_length` is 'Missing': Ask about length and recommend based on mood + genre + subgenre.
    - Else if `collected_directors` is 'Missing': Ask about directors and recommend based on previous preferences.
    - Else if `collected_actors` is 'Missing': Ask about actors and recommend based on previous preferences.
    - Else: Provide final recommendations based on all collected preferences.

**Guidelines for Each Step:**
- Use the preferences you have to provide 3-5 tailored movie recommendations before asking the next question
- For each movie, include TITLE, IMDb RATING (when available), and a BRIEF (1-2 sentence) justification
- Sort recommendations by IMDb rating when available
- Format recommendations clearly (e.g., using bullet points)
- When multiple preferences are collected at once, acknowledge them and move to the next missing preference

**Specific Questions to Ask:**
- For mood (if missing): "Hello! To recommend the perfect movie, what kind of mood are you in for a movie?"
- For genre (if mood collected): "Great! Now, what genre are you thinking of for a {collected_mood} movie?"
- For subgenre (if genre collected): "Great choices! Can you specify any subgenre within {collected_genre} that you particularly enjoy?"
- For length (if subgenre collected): "Do you prefer any specific length for your {collected_mood} {collected_genre} {collected_subgenre} movie?"
- For directors (if length collected): "Are there any specific directors whose work you would like to see in your {collected_mood} {collected_genre} {collected_subgenre} movie?"
- For actors (if directors collected): "Finally, are there any specific actors you wish to see in your movie?"

**Input Context:**
- `messages`: Full conversation history.
- `collected_mood`, `collected_genre`, `collected_subgenre`, `collected_length`, `collected_directors`, `collected_actors`: Preferences gathered so far.
- `{{context}}`: Database results based on latest query/topic.
- `{{web_results}}`: Web results based on latest query/topic.

--- START OF TURN INFO ---
Collected Mood: {collected_mood}
Collected Genre: {collected_genre}
Collected Subgenre: {collected_subgenre}
Collected Length: {collected_length}
Collected Directors: {collected_directors}
Collected Actors: {collected_actors}
Context from internal database (based on latest query/topic):
{{context}}

Web Search Results (based on latest query/topic):
{{web_results}}"""
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

def generate_response(state: MovieRecState) -> Dict:
    # Get context and web results, handle None cases
    context_docs = state.get("context", [])
    context_texts = [doc.page_content for doc in context_docs]
    formatted_context = "\n---\n".join(context_texts) if context_texts else "No internal context found."
    
    web_search_results = state.get("web_results", [])
    formatted_web_results = "\n---\n".join([json.dumps(res) for res in web_search_results]) if web_search_results else "No web search results found."

    # Parse user preferences from conversation
    preference_updates = parse_user_preferences(state)
    
    # Get conversation history and current collected state (including any updates)
    messages = state['messages']
    collected_mood = preference_updates.get("collected_mood", state.get("collected_mood"))
    collected_genre = preference_updates.get("collected_genre", state.get("collected_genre"))
    collected_subgenre = preference_updates.get("collected_subgenre", state.get("collected_subgenre"))
    collected_length = preference_updates.get("collected_length", state.get("collected_length"))
    collected_directors = preference_updates.get("collected_directors", state.get("collected_directors"))  # Added directors
    collected_actors = preference_updates.get("collected_actors", state.get("collected_actors"))          # Added actors

    # Create the chain for generation
    chain = RESPONSE_GENERATOR_PROMPT | llm | StrOutputParser()
    
    response_text = "Sorry, I encountered an error while generating the response."
    
    try:
        # Invoke the chain, passing current state info
        response_text = chain.invoke({
            "context": formatted_context,
            "web_results": formatted_web_results,
            "messages": messages, # Pass the history
            "collected_mood": collected_mood if collected_mood else "Missing",
            "collected_genre": collected_genre if collected_genre else "Missing",
            "collected_subgenre": collected_subgenre if collected_subgenre else "Missing",
            "collected_length": collected_length if collected_length else "Missing",
            "collected_directors": collected_directors if collected_directors else "Missing",  # Added directors
            "collected_actors": collected_actors if collected_actors else "Missing",          # Added actors
        })

    except Exception as e:
        print(f"Error during response generation: {e}")
        # Use default error message
        
    # Return the new AI message and any preference updates
    return {"messages": [AIMessage(content=response_text)], **preference_updates}

# --- Graph Nodes: Additional Nodes for Guided Start ---

FIRST_MOOD_QUESTION = "Hello! To recommend the perfect movie, what kind of mood are you in for a movie?"

def check_conversation_start(state: MovieRecState) -> Dict[str, str]:
    """Checks if this is the first user message and returns the next step key."""
    # The input to the graph is the user's query, making the history length 1 initially.
    if len(state['messages']) == 1:
        print("-- First turn, asking for mood.")
        return {"next_step": "ask_mood"}
    else:
        print("-- Continuing conversation, analyzing query.")
        return {"next_step": "analyze"}

def ask_mood_question(state: MovieRecState) -> Dict:
    """Returns the hardcoded first question asking for the user's mood."""
    return {"messages": [AIMessage(content=FIRST_MOOD_QUESTION)]}

# --- Graph Building with Dynamic Preference Handling ---
def build_graph():
    workflow = StateGraph(MovieRecState)
    
    # Add nodes
    workflow.add_node("check_start", check_conversation_start)
    workflow.add_node("ask_mood_initial", ask_mood_question)
    workflow.add_node("analyze_initial_input", analyze_initial_input)  # New node for initial preference detection
    workflow.add_node("analyze_query", analyze_query)
    workflow.add_node("adaptive_retrieval", adaptive_retrieval)
    workflow.add_node("web_search", web_search)
    workflow.add_node("generate_response", generate_response)

    # Set entry point
    workflow.set_entry_point("check_start")

    # Define conditional edges from the start check
    workflow.add_conditional_edges(
        "check_start",
        lambda state: state["next_step"],
        {
            "ask_mood": "analyze_initial_input",  # First check for initial preferences
            "analyze": "analyze_query",
        }
    )
    
    # After analyzing initial input, decide whether to ask for mood or proceed with response
    workflow.add_conditional_edges(
        "analyze_initial_input",
        lambda state: "generate_response" if state.get("collected_mood") else "ask_mood_initial",
        {
            "ask_mood_initial": "ask_mood_initial",
            "generate_response": "adaptive_retrieval",  # Skip mood question if already detected
        }
    )

    # Edge for the hardcoded mood question path
    workflow.add_edge("ask_mood_initial", END)

    # Define edges for the main conversational flow
    workflow.add_edge("analyze_query", "adaptive_retrieval")
    workflow.add_edge("adaptive_retrieval", "web_search")
    workflow.add_edge("web_search", "generate_response")
    workflow.add_edge("generate_response", END)
    
    return workflow

# --- Initialize Graph & Memory ---
workflow = build_graph()

memory = MemorySaver()

if vector_db is None:
    print("CRITICAL ERROR: Vector DB failed to initialize. Exiting.")
    exit() 

graph = workflow.compile(checkpointer=memory)

app = FastAPI(
    title="Movie Recommendation Agent API",
    description="API for interacting with a LangGraph-based movie recommendation agent."
)

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    config = {"configurable": {"thread_id": request.thread_id}}
    inputs = {"messages": [HumanMessage(content=request.query)]}
    final_response_message = "Sorry, I couldn't generate a response."

    try:
        for event in graph.stream(inputs, config=config, stream_mode="values"):
             final_state_event = event 

        if final_state_event and 'messages' in final_state_event:
             ai_messages = [msg for msg in final_state_event['messages'] if isinstance(msg, AIMessage)]
             if ai_messages:
                final_response_message = ai_messages[-1].content

    except Exception as e:
        print(f"Error in chat endpoint for thread {request.thread_id}: {e}")
        final_response_message = f"An error occurred: {e}"

    return ChatResponse(response=final_response_message)

if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)