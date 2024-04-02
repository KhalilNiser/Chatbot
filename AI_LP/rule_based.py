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
    # Default response
    return "I'm not sure how to respond to that. Can you try rephrasing?"


# Example chatbot conversation
user_message = input("You: ")
print("Bot:", get_response(user_message))
user_message = input("You: ")
print("Bot:", get_response(user_message))
