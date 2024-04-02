from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer, ChatterBotCorpusTrainer

import collections.abc
collections.Hashable = collections.abc.Hashable

# Create a new chatbot instance
chatbot = ChatBot('MyBot')

# Train the chatbot with a list of conversation examples
trainer = ChatterBotCorpusTrainer(chatbot)
trainer2_electric_boogaloo = ListTrainer(chatbot)
trainer.train(
    "chatterbot.corpus.english"
)

trainer2_electric_boogaloo.train([
    "Look at this, Mac. Look at this!",
    "What the hell is all this?",
    "This company is being bled like a stuck pig, Mac, and I've got a paper trail to prove it. Check this out.",
    "Look at this! That right there is the mail. Now let's talk about the mail. Can we talk about the mail, please, Mac? I've been dying to talk about the mail with you all day, OK? \"Pepe Silvia,\" this name keeps coming up over and over again. Every day Pepe's mail is getting sent back to me. Pepe Silvia! Pepe Silvia! I look in the mail, and this whole box is Pepe Silvia!",
])

# Start the conversation
while True:
    user_input = input('You: ')
    if user_input == 'Bye':
        break

    bot_response = chatbot.get_response(user_input)
    print('Bot:', bot_response)
