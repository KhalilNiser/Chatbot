# Define a function to handle greetings
def greet(user_input):
    greetings = ['hi', 'hello', 'hey', 'greetings']
    for greeting in greetings:
        if greeting in user_input.lower():
            return "Hello! How can I assist you today?"


# Define a function to handle farewells
def farewell(user_input):
    farewells = ['bye', 'goodbye', 'see you', 'later']
    for farewell in farewells:
        if farewell in user_input.lower():
            return "Goodbye! Have a great day!"


# Define a function to handle 'thank you'
def thank_you(user_input):
    if 'thank' in user_input.lower():
        return "You're welcome! Happy to help."

# Define a function to handle 'tell me a joke'
def joke(user_input):
    if 'tell me a joke' in user_input.lower():
        return "What did the comforter say after falling off the bed? Oh, sheet!"

# Define a function to handle 'what is the weather forecast'    
def weather(user_input):
    if 'what is the weather forecast' in user_input.lower():
        return "Go look outside, how would I know?"

# Define the main response function
def get_response(user_input):
    response = None
    # Check for greetings
    response = greet(user_input)
    if response:
        return response
    # Check for farewells
    response = farewell(user_input)
    if response:
        return response
    # Check for 'thank you'
    response = thank_you(user_input)
    if response:
        return response
    # Check for 'tell me a joke'
    response = joke(user_input)
    if response:
        return response
    # Check for 'what is the weather forecast'
    response = weather(user_input)
    if response:
        return response
    # Default response
    return "I'm not sure how to respond to that. Can you try rephrasing?"

# Example chatbot conversation
while True:
    user_input = input('You: ')
    if user_input == 'Bye':
        break

    bot_response = get_response(user_input)
    print('Bot:', bot_response)