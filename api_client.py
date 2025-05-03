import requests
import json

# --- Configuration ---
API_URL = "http://localhost:8000/chat"
# Use a consistent thread_id for a single conversation session
CONVERSATION_THREAD_ID = "client-session-alpha" 

print(f"Connecting to Movie Agent API at {API_URL}")
print(f"Using conversation thread ID: {CONVERSATION_THREAD_ID}")
print("Type 'quit' to exit.")

while True:
    try:
        user_input = input("\nYou: ")
        if user_input.lower() == 'quit':
            print("Exiting client.")
            break

        # Prepare the request payload
        payload = {
            "query": user_input,
            "thread_id": CONVERSATION_THREAD_ID
        }

        # Send POST request to the FastAPI endpoint
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        # Parse the JSON response
        response_data = response.json()
        
        # Print the assistant's response
        if "response" in response_data:
            print(f"Assistant: {response_data['response']}")
        else:
            print("Assistant: (Received an unexpected response format)")
            print(response_data)

    except requests.exceptions.ConnectionError:
        print("\nError: Could not connect to the API server.")
        print(f"Please ensure the FastAPI server (agent3.py) is running at {API_URL.rsplit('/', 1)[0]}")
        break # Exit if connection fails
    except requests.exceptions.RequestException as e:
        print(f"\nAn API request error occurred: {e}")
        # Optionally break or continue based on error type
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        break # Exit on other unexpected errors
