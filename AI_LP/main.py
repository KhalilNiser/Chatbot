from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer

# Create a new chatbot instance
chatbot = ChatBot('MyBot')

# Train the chatbot with a list of conversation examples
trainer = ListTrainer(chatbot)
trainer.train([
    'Hi there!',
    'Hello!',
    'How are you?',
    'I am doing well.',
    'What is your name?',
    'My name is MyBot.',
    'What can you do?',
    'I can chat with you about a variety of topics.',
])

# Start the conversation
while True:
    user_input = input('You: ')
    if user_input == 'Bye':
        break

    bot_response = chatbot.get_response(user_input)
    print('Bot:', bot_response)
