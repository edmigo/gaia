prev_topics = []

# User API
def user_login():
    # Authentication logic
    return {"message": "User logged in successfully"}


def get_models():
    # Retrieve model list allowed for the user
    models = {"names": ["Llama-2", "GAIA", "WizardCoder", "AURA" ,"ARAMS"],
              "URI" : [
                       "http://gaia-u-01.westeurope.cloudapp.azure.com:8080",
                       "http://127.0.0.1:8081/generate_stream",
                       "http://gaia-u-01.westeurope.cloudapp.azure.com:8090",
                       "http://127.0.0.1:8083/generate_stream", 
                       "http://127.0.0.1:8084/generate_stream"]}
    return models

def get_vectors():
    # Retrieve model list allowed for the user
    vectors = ["None", "AURA-Docs", "Erez-Docs", "MSDS-Docs"]
    return vectors

def get_embeddings():
    # Retrieve embedding list
    return {"names": ["john", "lock", "bond"]}

def get_welcome():
    # Generate a welcome message for the user using the LLM
    return {"message": "Welcome to the chat app!"}

def select_model():
    # Update the selected model for the user
    return {"message": "Model selected successfully"}

def set_creativity():
    # Set the required creativity level for the model
    return {"message": "Creativity level set successfully"}


# Chat API
def send_message():
    # Receive a user message and return the model's response
    return {"response": "Model response"}


def start_new_chat():
    # Clear the current chat context and start a new chat
    return {"message": "New chat started"}

def resume_chat():
    # Send the chat history for a resumed (older) chat
    return {"message": "Chat resumed successfully"}

def provide_feedback(feedback):
    # Log user feedback on the quality of the last response
    if feedback:
        print(feedback['type'])
        print(feedback['score'])
        print(feedback['text'])
        if feedback['score'] == '👍':
            print("positive feedback")
    return  


def get_chat_titles():
    # Get a title for the current active chat
    return ['Learn about signals and noise', 'What is the difference between a quantum computer and a supercomputer?', 'How to get a raise?']

def get_prev_chat_titles():
    # Get a title for the current active chat
    return {"title": "Chat Title"}

def get_follow_up_questions():
    # Retrieve a list of follow-up questions
    return {"questions": ["Question 1", "Question 2", "Question 3"]}


# Document API
def upload_document():
    # Upload a single PDF or text document to the server
    return {"message": "Document uploaded successfully"}


def clear_document():
    # Clear the current loaded file embedding from memory
    return {"message": "Document cleared successfully"}


# Embedding Set API
def get_embedding_sets():
    # Retrieve available embedding sets
    return {"sets": ["set1", "set2", "set3"]}

def select_embedding_set():
    # Select an embedding set for the user
    return {"message": "Embedding set selected successfully"}
