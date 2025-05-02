from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser

from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode
import os
from typing import TypedDict, List, Dict, Any, Optional

# Define the state schema
class AgentState(TypedDict):
    """State for the mood classification and content enrichment agent."""
    messages: List[Dict[str, Any]]
    mood: Optional[Dict[str, Any]]
    enriched_content: Optional[Dict[str, Any]]
    rephrased_content: Optional[str]
    entertainment_preferences: Optional[Dict[str, Any]]
    reflection_prompt: Optional[str]

# Initialize the Groq LLM
llm = ChatGroq(model="Llama3-8b-8192")

# System messages for different nodes
mood_classifier_system_msg = SystemMessage(
    content="""You are a mood and preference classifier. 
    Analyze the user's input and classify their mood and preferences.
    Return a JSON with the following structure:
    {
        "mood": "happy/sad/angry/neutral/romantic/anxious/etc.",
        "genre": "action/comedy/drama/thriller/horror/sci-fi/romance/etc.",
        "emotional_tone": "positive/negative/mixed/neutral",
        "energy_level": "low/medium/high",
        "preferences": ["preference1", "preference2", ...],
        "topics_of_interest": ["topic1", "topic2", ...]
    }"""
)

metadata_enricher_system_msg = SystemMessage(
    content="""You are a metadata enricher.
    Based on the user's input and their classified mood and preferences,
    enrich the content with additional metadata.
    Return a JSON with the following structure:
    {
        "key_entities": ["entity1", "entity2", ...],
        "sentiment_analysis": {
            "positive": 0.0-1.0,
            "negative": 0.0-1.0,
            "neutral": 0.0-1.0
        },
        "suggested_actions": ["action1", "action2", ...],
        "content_categories": ["category1", "category2", ...],
        "emotional_triggers": ["trigger1", "trigger2", ...]
    }"""
)

content_rephraser_system_msg = SystemMessage(
    content="""You are a content rephraser.
    Based on the user's input, their classified mood, and the enriched metadata,
    rephrase the user's message in a way that aligns with their mood and preferences.
    Make the rephrased content more engaging, personalized, and contextually relevant.
    """
)

entertainment_preferences_system_msg = SystemMessage(
    content="""You are an entertainment preference analyzer.
    Based on the user's input, their mood, and the enriched metadata,
    extract or infer their entertainment preferences.
    Return a JSON with the following structure:
    {
        "favorite_actors": ["actor1", "actor2", ...] or null if unknown,
        "favorite_movies": ["movie1", "movie2", ...] or null if unknown,
        "genre_preferences": ["genre1", "genre2", ...] or ["unknown"],
        "language_preferences": ["language1", "language2", ...] or ["unknown"],
        "region_preferences": ["region1", "region2", ...] or ["unknown"],
        "time_availability": "short film/series/2-hour movie" or "unknown",
        "mood_based_recommendations": ["recommendation1", "recommendation2", ...]
    }
    
    Be creative but realistic in your inferences. If the user hasn't mentioned something specific,
    make educated guesses based on their mood and other preferences, but mark clearly what is inferred vs. explicitly stated.
    """
)

reflection_prompt_system_msg = SystemMessage(
    content="""You are a reflection prompt generator.
    Based on the user's mood, preferences, and the enriched metadata,
    create a thoughtful prompt that encourages the user to reflect on and share more about their feelings.
    The prompt should be personalized, empathetic, and relevant to their current emotional state.
    Keep it brief (2-3 sentences) but impactful.
    """
)

# Node functions
def classify_mood(state: AgentState) -> AgentState:
    """Classify the user's mood and preferences."""
    # Get the latest user message
    user_message = state["messages"][-1]
    
    # Create a prompt for mood classification
    json_parser = JsonOutputParser()
    
    # Use the LLM to classify mood
    response = llm.invoke([
        mood_classifier_system_msg,
        HumanMessage(content=f"Classify the mood and preferences in this message: {user_message['content'] if isinstance(user_message, dict) else user_message}")
    ])
    
    # Parse the response as JSON
    try:
        mood_data = json_parser.invoke(response.content)
    except:
        # Fallback if JSON parsing fails
        mood_data = {
            "mood": "neutral",
            "genre": "drama",
            "emotional_tone": "neutral",
            "energy_level": "medium",
            "preferences": [],
            "topics_of_interest": []
        }
    
    # Update the state with mood classification
    return {"messages": state["messages"], "mood": mood_data}

def enrich_metadata(state: AgentState) -> AgentState:
    """Enrich the content with additional metadata."""
    # Get the latest user message and mood classification
    user_message = state["messages"][-1]
    mood_data = state["mood"]
    
    # Create a prompt for metadata enrichment
    json_parser = JsonOutputParser()
    
    # Use the LLM to enrich metadata
    response = llm.invoke([
        metadata_enricher_system_msg,
        HumanMessage(content=f"""
        User message: {user_message['content'] if isinstance(user_message, dict) else user_message}
        Mood classification: {mood_data}
        
        Enrich this content with additional metadata.
        """)
    ])
    
    # Parse the response as JSON
    try:
        enriched_data = json_parser.invoke(response.content)
    except:
        # Fallback if JSON parsing fails
        enriched_data = {
            "key_entities": [],
            "sentiment_analysis": {"positive": 0.33, "negative": 0.33, "neutral": 0.34},
            "suggested_actions": [],
            "content_categories": [],
            "emotional_triggers": []
        }
    
    # Update the state with enriched metadata
    return {
        "messages": state["messages"], 
        "mood": state["mood"], 
        "enriched_content": enriched_data
    }

def analyze_entertainment_preferences(state: AgentState) -> AgentState:
    """Analyze entertainment preferences based on user input and mood."""
    # Get the latest user message, mood, and enriched metadata
    user_message = state["messages"][-1]
    mood_data = state["mood"]
    enriched_data = state["enriched_content"]
    
    # Create a parser for entertainment preferences
    json_parser = JsonOutputParser()
    
    # Use the LLM to analyze entertainment preferences
    response = llm.invoke([
        entertainment_preferences_system_msg,
        HumanMessage(content=f"""
        User message: {user_message['content'] if isinstance(user_message, dict) else user_message}
        Mood classification: {mood_data}
        Enriched metadata: {enriched_data}
        
        Extract or infer the user's entertainment preferences.
        """)
    ])
    
    # Parse the response as JSON
    try:
        preferences_data = json_parser.invoke(response.content)
    except:
        # Fallback if JSON parsing fails
        preferences_data = {
            "favorite_actors": None,
            "favorite_movies": None,
            "genre_preferences": ["unknown"],
            "language_preferences": ["unknown"],
            "region_preferences": ["unknown"],
            "time_availability": "unknown",
            "mood_based_recommendations": []
        }
    
    # Update the state with entertainment preferences
    return {
        "messages": state["messages"],
        "mood": state["mood"],
        "enriched_content": state["enriched_content"],
        "entertainment_preferences": preferences_data
    }

def generate_reflection_prompt(state: AgentState) -> AgentState:
    """Generate a reflection prompt for the user to share more about their feelings."""
    # Get the latest user message, mood, and enriched metadata
    user_message = state["messages"][-1]
    mood_data = state["mood"]
    enriched_data = state["enriched_content"]
    entertainment_prefs = state["entertainment_preferences"]
    
    # Use the LLM to generate a reflection prompt
    response = llm.invoke([
        reflection_prompt_system_msg,
        HumanMessage(content=f"""
        User message: {user_message['content'] if isinstance(user_message, dict) else user_message}
        Mood classification: {mood_data}
        Enriched metadata: {enriched_data}
        Entertainment preferences: {entertainment_prefs}
        
        Generate a thoughtful reflection prompt for the user.
        """)
    ])
    
    # Update the state with the reflection prompt
    return {
        "messages": state["messages"],
        "mood": state["mood"],
        "enriched_content": state["enriched_content"],
        "entertainment_preferences": state["entertainment_preferences"],
        "reflection_prompt": response.content
    }

def rephrase_content(state: AgentState) -> AgentState:
    """Rephrase the user's content based on mood and metadata."""
    # Get the latest user message, mood, and enriched metadata
    user_message = state["messages"][-1]
    mood_data = state["mood"]
    enriched_data = state["enriched_content"]
    entertainment_prefs = state["entertainment_preferences"]
    reflection_prompt = state["reflection_prompt"]
    
    # Use the LLM to rephrase content
    response = llm.invoke([
        content_rephraser_system_msg,
        HumanMessage(content=f"""
        Original message: {user_message['content'] if isinstance(user_message, dict) else user_message}
        Mood classification: {mood_data}
        Enriched metadata: {enriched_data}
        Entertainment preferences: {entertainment_prefs}
        
        Rephrase the original message in a way that aligns with the user's mood and preferences.
        """)
    ])
    
    # Update the state with rephrased content
    return {
        "messages": state["messages"], 
        "mood": state["mood"], 
        "enriched_content": state["enriched_content"],
        "entertainment_preferences": state["entertainment_preferences"],
        "reflection_prompt": state["reflection_prompt"],
        "rephrased_content": response.content
    }

# Function to process user input
def process_user_input(user_input: str):
    """Process user input through the agent workflow."""
    # Initialize the agent
    agent = build_agent()
    
    # Create initial state with user message
    initial_state = {
        "messages": [{"role": "user", "content": user_input}],
        "mood": None,
        "enriched_content": None,
        "entertainment_preferences": None,
        "reflection_prompt": None,
        "rephrased_content": None
    }
    
    # Run the agent
    result = agent.invoke(initial_state)
    
    return {
        "original_input": user_input,
        "mood_classification": result["mood"],
        "enriched_metadata": result["enriched_content"],
        "entertainment_preferences": result["entertainment_preferences"],
        "reflection_prompt": result["reflection_prompt"],
        "rephrased_content": result["rephrased_content"]
    }

# Build the graph
def build_agent():
    """Build and return the agent graph."""
    # Create a new graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("classify_mood", classify_mood)
    workflow.add_node("enrich_metadata", enrich_metadata)
    workflow.add_node("analyze_entertainment_preferences", analyze_entertainment_preferences)
    workflow.add_node("generate_reflection_prompt", generate_reflection_prompt)
    workflow.add_node("rephrase_content", rephrase_content)
    
    # Add edges
    workflow.add_edge(START, "classify_mood")
    workflow.add_edge("classify_mood", "enrich_metadata")
    workflow.add_edge("enrich_metadata", "analyze_entertainment_preferences")
    workflow.add_edge("analyze_entertainment_preferences", "generate_reflection_prompt")
    workflow.add_edge("generate_reflection_prompt", "rephrase_content")
    workflow.add_edge("rephrase_content", END)
    
    # Compile the graph
    return workflow.compile()

# Example usage
if __name__ == "__main__":
    # Get user input
    user_input = input("Enter your message: ")
    
    # Process the input
    result = process_user_input(user_input)
    
    # Display the results
    print("\nMood Classification:")
    print(result["mood_classification"])
    
    print("\nEnriched Metadata:")
    print(result["enriched_metadata"])
    
    print("\nEntertainment Preferences:")
    print(result["entertainment_preferences"])
    
    print("\nReflection Prompt:")
    print(result["reflection_prompt"])
    
    print("\nRephrased Content:")
    print(result["rephrased_content"])
    
    # Ask for reflection
    print("\n" + "=" * 50)
    print(result["reflection_prompt"])
    reflection = input("\nYour thoughts (2-3 lines about your feelings): ")
    
    if reflection:
        print("\nThank you for sharing your feelings!")

# Build and export the graph for LangGraph runtime
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("classify_mood", classify_mood)
workflow.add_node("enrich_metadata", enrich_metadata)
workflow.add_node("analyze_entertainment_preferences", analyze_entertainment_preferences)
workflow.add_node("generate_reflection_prompt", generate_reflection_prompt)
workflow.add_node("rephrase_content", rephrase_content)

# Add edges
workflow.add_edge(START, "classify_mood")
workflow.add_edge("classify_mood", "enrich_metadata")
workflow.add_edge("enrich_metadata", "analyze_entertainment_preferences")
workflow.add_edge("analyze_entertainment_preferences", "generate_reflection_prompt")
workflow.add_edge("generate_reflection_prompt", "rephrase_content")
workflow.add_edge("rephrase_content", END)

# Compile the graph and export it
graph = workflow.compile()
