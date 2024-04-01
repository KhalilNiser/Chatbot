from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer, ChatterBotCorpusTrainer

import collections.abc
collections.Hashable = collections.abc.Hashable

# Create a new chatbot instance
chatbot = ChatBot('MyBot')

# Train the chatbot with a list of conversation examples
trainer = ChatterBotCorpusTrainer(chatbot)
trainer.train(
    "chatterbot.corpus.english"
)

# Start the conversation
while True:
    user_input = input('You: ')
    if user_input == 'Bye':
        break

    bot_response = chatbot.get_response(user_input)
    print('Bot:', bot_response)
