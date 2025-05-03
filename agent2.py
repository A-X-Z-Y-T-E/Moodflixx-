from typing import TypedDict, List, Dict, Literal, Optional, Annotated, Union
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.runnables.config import RunnableConfig
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END, add_messages
import pandas as pd
import os
import json
import ast
import pickle
import torch
from dotenv import load_dotenv
from pydantic.v1 import BaseModel, Field
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq

load_dotenv()

# Define the state for our movie recommendation system
class MovieRecState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    context: Annotated[Optional[List[Document]], lambda x, y: y]
    query: Annotated[Optional[str], lambda x, y: y]
    response: Annotated[Optional[str], lambda x, y: y]
    retrieval_type: Annotated[Optional[str], lambda x, y: y]
    web_results: Annotated[Optional[List[Dict]], lambda x, y: y]

# Initialize the LLM with Groq
api_key = os.environ.get("GROQ_API_KEY")
llm = ChatGroq(model_name="llama3-70b-8192", api_key=api_key)

# Check if GPU is available and initialize embeddings model
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize embeddings model with GPU if available
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": device}
)

# Function to create vector database from movie data
def create_vector_db(csv_path='data/processed/processed_movies.csv'):
    # Define paths for FAISS index and pickle file
    faiss_index_path = "data/vectordb/faiss_index"
    pickle_path = "data/vectordb/faiss_documents.pkl"
    
    # Create directory if it doesn't exist
    if not os.path.exists("data/vectordb"):
        os.makedirs("data/vectordb", exist_ok=True)
    
    # Check if vector DB already exists
    if os.path.exists(faiss_index_path) and os.path.exists(pickle_path):
        # Load existing vector DB
        try:
            with open(pickle_path, "rb") as f:
                stored_docs = pickle.load(f)
            vectordb = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
            return vectordb
        except Exception as e:
            print(f"Error loading existing vector DB: {e}")
            print("Creating a new vector DB...")
            # If loading fails, we'll create a new one
    
    # Load the movies dataset from CSV
    movies_df = pd.read_csv(csv_path)
    
    # Create documents for vector DB
    documents = []
    
    for idx, row in movies_df.iterrows():
        try:
            # Create a comprehensive text representation of the movie
            movie_text = f"Title: {row['movie_title'] if not pd.isna(row['movie_title']) else 'Unknown'}\n"
            movie_text += f"Year: {int(row['title_year']) if not pd.isna(row['title_year']) else 'Unknown'}\n"
            movie_text += f"Genres: {row['genres'] if not pd.isna(row['genres']) else 'Unknown'}\n"
            movie_text += f"Director: {row['director_name'] if not pd.isna(row['director_name']) else 'Unknown'}\n"
            movie_text += f"Rating: {row['imdb_score'] if not pd.isna(row['imdb_score']) else 'Unknown'}/10\n"
            movie_text += f"Actors: {row['actor_1_name'] if not pd.isna(row['actor_1_name']) else 'Unknown'}, "
            movie_text += f"{row['actor_2_name'] if not pd.isna(row['actor_2_name']) else 'Unknown'}, "
            movie_text += f"{row['actor_3_name'] if not pd.isna(row['actor_3_name']) else 'Unknown'}\n"
            movie_text += f"Plot Keywords: {row['plot_keywords'] if not pd.isna(row['plot_keywords']) else 'Unknown'}\n"
            movie_text += f"Plot Summary: {row['plot_summary'] if not pd.isna(row['plot_summary']) else 'Unknown'}\n"
            
            # Create document with metadata
            metadata = {
                'id': int(idx),
                'title': row['movie_title'] if not pd.isna(row['movie_title']) else 'Unknown',
                'year': int(row['title_year']) if not pd.isna(row['title_year']) else None,
                'genres': row['genres'] if not pd.isna(row['genres']) else 'Unknown',
                'rating': float(row['imdb_score']) if not pd.isna(row['imdb_score']) else None,
                'director': row['director_name'] if not pd.isna(row['director_name']) else 'Unknown'
            }
            
            documents.append(Document(page_content=movie_text, metadata=metadata))
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
    
    # Create FAISS index
    vectordb = FAISS.from_documents(documents, embeddings)
    
    # Save the FAISS index and documents
    vectordb.save_local(faiss_index_path)
    with open(pickle_path, "wb") as f:
        pickle.dump(documents, f)
    
    return vectordb

# Create or load vector DB - Lazy initialization to avoid errors when importing the module
# We'll initialize it only when needed
vectordb = None

def get_vectordb():
    global vectordb
    if vectordb is None:
        vectordb = create_vector_db()
    return vectordb

# Create basic retriever - initialize only when needed
def get_retriever():
    return get_vectordb().as_retriever(search_kwargs={"k": 5})

# Create compressor for contextual compression
compressor = LLMChainExtractor.from_llm(llm)

# Create compression retriever - initialize only when needed
def get_compression_retriever():
    return ContextualCompressionRetriever(
        base_retriever=get_retriever(),
        base_compressor=compressor
    )

# Define the RAG prompt
rag_prompt = ChatPromptTemplate.from_template(
    """You are a helpful movie recommendation assistant. 
    Your goal is to recommend 5 to 10 movies based on the user's request and the provided context.
    
    Analyze the user's question and the context from both our movie database and the web search results. 
    Synthesize this information to provide a list of relevant movie recommendations.

    Context from our movie database:
    ---------------------
    {context}
    ---------------------
    Relevant Web Search Results:
    ---------------------
    {web_context}
    ---------------------
    User Question: {question}
    
    Based on all the above, please recommend between 5 and 10 movies that fit the user's request. List the titles clearly.
    Answer:"""
)

# Define the query analysis prompt
query_analyzer_prompt = ChatPromptTemplate.from_messages([
    ("system", """Analyze the user's query about movies and determine the best retrieval approach.
    
    Options:
    - "semantic": For general questions about movie recommendations, themes, or when the user is exploring.
    - "keyword": For specific questions about particular movies, directors, actors, or years.
    - "none": If no retrieval is needed (e.g., greetings, thanks, or general conversation).
    
    Return ONLY the chosen option as a single word: "semantic", "keyword", or "none".
    """),
    ("human", "{query}")
])

# Define the adaptive retrieval function
def adaptive_retrieval(state: MovieRecState) -> Dict:
    # Get the last message from the user
    if not state["messages"]:
        # Return empty context directly
        return {"context": []}

    # Get the last message and extract content safely
    last_message = state["messages"][-1]

    # Check if it's a dict (from API) or a BaseMessage object
    if isinstance(last_message, dict) and "content" in last_message:
        query = last_message["content"]
    elif isinstance(last_message, BaseMessage):
        query = last_message.content
    else:
        # Fallback if we can't get content
        query = str(last_message)

    # Analyze the query to determine retrieval approach
    # Build the analysis chain
    analysis_chain = query_analyzer_prompt | llm | StrOutputParser()
    # Invoke the chain
    retrieval_type = analysis_chain.invoke({"query": query})
    # Store retrieval_type locally, don't modify state directly yet
    current_retrieval_type = retrieval_type.strip().lower()

    # Perform retrieval based on the determined approach
    # Always use the basic retriever if retrieval is needed
    context_update = {} # Initialize update dict
    query_update = {"query": query} # Prepare query update
    retrieval_type_update = {"retrieval_type": current_retrieval_type} # Prepare type update

    if current_retrieval_type != "none":
        retrieved_docs = get_retriever().get_relevant_documents(query)
        context_update = {"context": retrieved_docs}
    else:
        context_update = {"context": []}

    # Return all updates this node is responsible for
    return {**query_update, **retrieval_type_update, **context_update}

# Define the web search function
def web_search(state: MovieRecState) -> Dict: # Return type changed to Dict
    # Get the query from the state (set by adaptive_retrieval)
    query = state.get("query")
    if not query:
        return {"web_results": []}

    # Perform web search
    tavily_tool = TavilySearchResults(max_results=3)
    web_results = tavily_tool.invoke({"query": query})
    
    # Return only the updated key
    return {"web_results": web_results}

# Define the generation function
def generate_response(state: MovieRecState) -> Dict: # Return type changed to Dict
    # Format context for the prompt
    context_docs = state.get("context") # Get context, might be None
    context_texts = [doc.page_content for doc in context_docs] if context_docs else [] # Check if None before iterating
    formatted_context = "\n\n".join(context_texts) if context_texts else "No relevant movie information found."
    
    # Format web results
    web_results = state.get("web_results", [])
    formatted_web_results = "\n\n".join([str(res) for res in web_results]) if web_results else "No relevant web search results found."
    
    # Get the user's question
    user_question = state["query"]
    
    # Create the prompt
    prompt = ChatPromptTemplate.from_template(
        """You are a helpful movie recommendation assistant. 
        Your goal is to recommend 5 to 10 movies based on the user's request and the provided context.
        
        Analyze the user's question and the context from both our movie database and the web search results. 
        Synthesize this information to provide a list of relevant movie recommendations.

        Context from our movie database:
        ---------------------
        {context}
        ---------------------
        Relevant Web Search Results:
        ---------------------
        {web_context}
        ---------------------
        User Question: {question}
        
        Based on all the above, please recommend between 5 and 10 movies that fit the user's request. List the titles clearly.
        Answer:"""
    )
    
    # Create the generation chain
    llm_chain = prompt | llm | StrOutputParser()
    
    # Generate the response
    response = llm_chain.invoke({
        "context": formatted_context,
        "web_context": formatted_web_results,
        "question": user_question
    })

    # Return only the new message for add_messages to handle
    # response is already a string due to StrOutputParser
    return {"messages": [AIMessage(content=response)]}

# Define the router function
def router(state: MovieRecState) -> Literal["retrieve", "respond", "end"]:
    # If no messages, start with retrieval
    if not state["messages"]:
        return "retrieve"
    
    # If last message is from AI, we're done with this round
    if len(state["messages"]) > 0 and isinstance(state["messages"][-1], AIMessage):
        return "end"
    
    # Check if user wants to end conversation
    last_message = state["messages"][-1]
    if isinstance(last_message, dict) and "content" in last_message:
        content = last_message["content"].lower()
    elif isinstance(last_message, BaseMessage):
        content = last_message.content.lower()
    else:
        content = str(last_message).lower()
        
    if "thank" in content or "bye" in content or "thanks" in content:
        return "end"
    
    # Otherwise, proceed with retrieval
    return "retrieve"

# Define the analyze query function
def analyze_query(state: MovieRecState) -> Dict:
    # Get the last message from the user
    if not state["messages"]:
        # Handle case where there are no messages
        return {"query": "", "retrieval_type": "none"}

    # Get the last message and extract content safely
    last_message = state["messages"][-1]
    
    # Check if it's a dict (from API) or a BaseMessage object
    if isinstance(last_message, dict) and "content" in last_message:
        query = last_message["content"]
    elif isinstance(last_message, BaseMessage):
        query = last_message.content
    else:
        # Fallback if we can't get content
        query = str(last_message)
    
    # Set the query
    query_update = {"query": query}
    
    # Analyze the query to determine retrieval approach
    # Build the analysis chain
    analysis_chain = query_analyzer_prompt | llm | StrOutputParser()
    # Invoke the chain
    retrieval_type = analysis_chain.invoke({"query": query})
    retrieval_type_update = {"retrieval_type": retrieval_type.strip().lower()}
    
    return {**query_update, **retrieval_type_update}

# Create the graph
def build_graph():
    workflow = StateGraph(MovieRecState)
    
    workflow.add_node("analyze_query", analyze_query)
    workflow.add_node("adaptive_retrieval", adaptive_retrieval)
    workflow.add_node("web_search", web_search) 
    workflow.add_node("generate_response", generate_response)
    
    workflow.set_entry_point("analyze_query")
    
    workflow.add_edge("analyze_query", "adaptive_retrieval")
    workflow.add_edge("analyze_query", "web_search") 
    
    workflow.add_edge("adaptive_retrieval", "generate_response")
    workflow.add_edge("web_search", "generate_response") 
    
    workflow.add_edge("generate_response", END)
    
    return workflow.compile()

# Create and export the graph for LangGraph Studio
# This needs to be a concrete graph instance, not a function
graph = build_graph()

try:
    # Use draw_mermaid_png() instead of draw_png()
    graph_image = graph.get_graph().draw_mermaid_png() 
    with open("workflow_graph.png", "wb") as f:
        f.write(graph_image)
except Exception as e:
    print(f"An error occurred while saving the graph: {e}")

# For testing purposes
if __name__ == "__main__":
    # Initialize state with a test message
    state: MovieRecState = {
        "messages": [HumanMessage(content="I'm looking for action movies with high ratings")],
        "context": None,
        "query": None,
        "response": None,
        "retrieval_type": None,
        "web_results": []
    }
    
    # Process request
    state = graph.invoke(state)
    
    # Print response
    if state["messages"] and isinstance(state["messages"][-1], AIMessage):
        print(f"Assistant: {state['messages'][-1].content}")