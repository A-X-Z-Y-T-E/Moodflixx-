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

RESPONSE_GENERATOR_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful and knowledgeable movie recommendation assistant. Use the provided context from our internal database and external web search results to answer the user's query."
            "Prioritize information from the internal database (context) if available and relevant. Supplement with web search results for broader information, recent movies, or details not in the database."
            "If the query is conversational (e.g., 'hello'), respond naturally."
            "Combine the information thoughtfully. If context and web results conflict, state that or prioritize based on likely user intent (e.g., prefer database plot summaries but web results for recent ratings)."
            "Always be friendly and aim to provide 3-5 relevant movie recommendations unless the user asks for a different number."
            "Format your response clearly."
            "Context from internal database:\n{context}\n\nWeb Search Results:\n{web_results}"
        ),
        MessagesPlaceholder(variable_name="messages"), 
    ]
)

def generate_response(state: MovieRecState) -> Dict:
    context_docs = state.get("context", [])
    context_texts = [doc.page_content for doc in context_docs]
    formatted_context = "\n---\n".join(context_texts) if context_texts else "No internal context found."
    
    web_search_results = state.get("web_results", [])
    formatted_web_results = "\n---\n".join([json.dumps(res) for res in web_search_results]) if web_search_results else "No web search results found."

    messages = state['messages']
    
    chain = RESPONSE_GENERATOR_PROMPT | llm | StrOutputParser()
    
    try:
        response = chain.invoke({
            "context": formatted_context,
            "web_results": formatted_web_results,
            "messages": messages 
        })
    except Exception as e:
        print(f"Error during response generation: {e}")
        response = "Sorry, I encountered an error while generating the response."
        
    return {"messages": [AIMessage(content=response)]}

def build_graph():
    workflow = StateGraph(MovieRecState)
    
    workflow.add_node("analyze_query", analyze_query)
    workflow.add_node("adaptive_retrieval", adaptive_retrieval)
    workflow.add_node("web_search", web_search)
    workflow.add_node("generate_response", generate_response)
    
    workflow.set_entry_point("analyze_query")
    workflow.add_edge("analyze_query", "adaptive_retrieval")
    workflow.add_edge("adaptive_retrieval", "web_search")
    workflow.add_edge("web_search", "generate_response")
    workflow.add_edge("generate_response", END)
    
    return workflow

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