import spacy

# Load the spaCy model
nlp = spacy.load('en_core_web_sm')

# Define some intents and responses
intents = {
    'greet': ['Hello', 'Hi', 'Hey'],
    'goodbye': ['Bye', 'Goodbye', 'See you later'],
    'thankyou': ['Thanks', 'Thank you', 'Appreciate it']
}

responses = {
    'greet': 'Hello! How can I help you today?',
    'goodbye': 'Goodbye! Have a great day!',
    'thankyou': 'You\'re welcome! Happy to help!'
}


# Function to find the intent of the message
def find_intent(message):
    # Process the message with spaCy
    doc = nlp(message)
    # Calculate similarity with each intent
    scores = {intent: doc.similarity(nlp(' '.join(phrases))) for intent, phrases in intents.items()}
    # Find the intent with the highest score
    return max(scores, key=scores.get)


# Function to respond to a message
def respond(message):
    intent = find_intent(message)
    return responses[intent]


# Example conversation
print(respond('Hi there!'))  # Should respond with a greeting
print(respond('Thanks for your help'))  # Should respond with a thank you response
