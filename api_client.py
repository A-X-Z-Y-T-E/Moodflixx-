import requests
import json
import uuid

# --- Configuration ---
API_URL = "http://localhost:8000/chat"
# Use a consistent thread_id for a single conversation session
CONVERSATION_THREAD_ID = f"client-session-{uuid.uuid4().hex[:6]}"

print(f"Connecting to Movie Agent API at {API_URL}")
print(f"Using conversation thread ID: {CONVERSATION_THREAD_ID}")
print("Type 'quit' to exit.")
print("\n-- Starting Conversation --")

def make_api_request(query, thread_id):
    """Sends a request to the FastAPI endpoint and returns the response."""
    try:
        response = requests.post(API_URL, json={"query": query, "thread_id": thread_id})
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        return response.json().get("response", "Error: No response field found.")
    except requests.exceptions.ConnectionError:
        print(f"\nError: Could not connect to the API server.")
        print(f"Please ensure the FastAPI server (agent3.py) is running at {API_URL.replace('/chat','')}")
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"\nError during API request: {e}")
        return f"Error communicating with API: {e}"

# --- Initial Request to get the first prompt (mood question) ---
try:
    initial_response = make_api_request("hello", CONVERSATION_THREAD_ID) # Send a dummy first message
    print(f"\nAssistant: {initial_response}")
except SystemExit: # Catch exit from connection error
    pass # Error message already printed
except Exception as e:
    print(f"\nUnexpected error during initial connection: {e}")
    sys.exit(1)

# --- Main Conversation Loop ---
while True:
    try:
        user_input = input("\nYou: ")
        if user_input.lower() == 'quit':
            print("Exiting chat.")
            break

        if not user_input:
            continue

        assistant_response = make_api_request(user_input, CONVERSATION_THREAD_ID)
        print(f"\nAssistant: {assistant_response}")

    except EOFError:
        # Handle Ctrl+D or other EOF signals gracefully
        print("\nExiting chat due to EOF.")
        break
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        print("\nExiting chat due to interrupt.")
        break
    except Exception as e:
        # Catch any other unexpected errors during the loop
        print(f"\nAn unexpected error occurred: {e}")
        # Optionally, you might want to break or continue based on the error
        break
