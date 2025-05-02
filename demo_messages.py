"""
Demo messages for testing the mood classification and content enrichment agent.
This file contains a variety of example messages with different moods, preferences, and content.
"""

from agent import process_user_input
import json

# List of demo messages with different moods and content
DEMO_MESSAGES = [
    # Happy/Positive Messages with Entertainment References
    "I just got a promotion at work! I'm so excited and want to celebrate by watching a good comedy movie tonight, maybe something with Ryan Reynolds.",
    "The weather is beautiful today! I'm planning to go hiking with friends and then maybe watch a thriller movie in the evening. I love movies like Inception.",
    
    # Sad/Negative Messages with Entertainment References
    "I've been feeling down lately. I usually watch romantic movies like The Notebook to cheer me up, but I'm open to other suggestions.",
    "My pet has been sick, and I'm really worried. I need something to distract me, maybe a good series on Netflix? I prefer Hindi content.",
    
    # Mixed Emotions with Entertainment References
    "I passed my exam but my friend didn't. I feel both happy and sad. Maybe a drama movie would match my mood? I have about 2 hours free tonight.",
    "I'm excited about my vacation next week, but anxious about all the work I need to finish. I enjoy Tamil movies with good storylines when I'm stressed.",
    
    # Messages with Clear Entertainment Preferences
    "I prefer action movies with lots of special effects and a good storyline. Tom Cruise and Keanu Reeves are my favorite actors.",
    "I'm looking for a good Telugu movie to watch this weekend. I enjoy thrillers and have about 3 hours free.",
    "I enjoy watching horror movies late at night, especially Japanese ones. Any recommendations for a short film in this genre?",
    
    # Messages with Mixed Preferences
    "I'm in a romantic mood today. I love watching Shah Rukh Khan movies, especially the romantic ones. But I also enjoy thrillers occasionally.",
    "I'm feeling anxious about my upcoming presentation. I usually watch comedy series to relax. I have about 30 minutes free now."
]

def pretty_print_json(data):
    """Print JSON data in a formatted way."""
    if data is None:
        print("None")
        return
    
    try:
        if isinstance(data, str):
            # Try to parse as JSON if it's a string
            data = json.loads(data)
        print(json.dumps(data, indent=2))
    except:
        # If it's not valid JSON, just print it as is
        print(data)

def run_demo():
    """Run the agent on all demo messages and display the results."""
    print("=" * 80)
    print("MOOD CLASSIFICATION AND ENTERTAINMENT PREFERENCES DEMO")
    print("=" * 80)
    
    for i, message in enumerate(DEMO_MESSAGES):
        print(f"\nDEMO MESSAGE #{i+1}:")
        print(f"Original: {message}")
        
        try:
            # Process the message
            result = process_user_input(message)
            
            # Display results
            print("\nMood Classification:")
            pretty_print_json(result["mood_classification"])
            
            print("\nEnriched Metadata:")
            pretty_print_json(result["enriched_metadata"])
            
            print("\nEntertainment Preferences:")
            pretty_print_json(result["entertainment_preferences"])
            
            print("\nReflection Prompt:")
            print(result["reflection_prompt"])
            
            print("\nRephrased Content:")
            print(result["rephrased_content"])
            
            # Simulate user reflection
            print("\n" + "=" * 50)
            print(f"[Simulated User Reflection based on prompt: {result['reflection_prompt']}]")
            print("I would write about my feelings here if this wasn't a demo...")
            
        except Exception as e:
            print(f"\nError processing message: {str(e)}")
        
        print("\n" + "=" * 80)
        
        # Ask if user wants to continue to the next message
        if i < len(DEMO_MESSAGES) - 1:
            response = input("Press Enter to continue to the next message (or 'q' to quit): ")
            if response.lower() == 'q':
                break

def test_single_message(message):
    """Test the agent with a single message."""
    try:
        print(f"Original: {message}")
        
        # Process the message
        result = process_user_input(message)
        
        # Display results
        print("\nMood Classification:")
        pretty_print_json(result["mood_classification"])
        
        print("\nEnriched Metadata:")
        pretty_print_json(result["enriched_metadata"])
        
        print("\nEntertainment Preferences:")
        pretty_print_json(result["entertainment_preferences"])
        
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
            
    except Exception as e:
        print(f"\nError processing message: {str(e)}")

if __name__ == "__main__":
    # Choose a specific message to test or run all demos
    single_test = False
    
    if single_test:
        # Test with a single message
        test_message = "I'm feeling happy today because I just got a new job offer! I might celebrate by watching a good movie tonight. I love action films with Tom Cruise, but I'm also in the mood for something funny."
        test_single_message(test_message)
    else:
        # Run all demo messages
        run_demo()
