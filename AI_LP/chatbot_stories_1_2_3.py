# ---- IMPORT_REQUIRED_LIBRARIES ----
import random
from flask import Flask, request, jsonify, render_template 
# IMPORT chatbot from ChatBot folder
from chatbot import VRChatbot
import os


app = Flask(__name__, template_folder=os.path.join("ChatBot", "templates"))


# ---- CREATE AN INSTANCE OF "chatbot" ----
chatbot = VRChatbot



# ---- DEFINE_CLASS_VRChatbot ----
class VRChatbot:
    # ---- INITIALIZING THE Chatbot __Init__ METHOD(): ----
    # __Init__: Special method (Constructor) that is called when an
    # object of the class is created. 
    # Self: Allows access to all atrributes and data fields of the class
    def __init__(self):
        # Different phrases/ways a user might greet the "chatbot"
        self.greetings = ["hello", "hi!", "hey", "greetings", "what's up", "howdy", "good day"]
        # different phrases/ways a user might end the conversation
        self.farewells = ["goodbye", "bye", "exit", "end chat", "close conversation", "see you later", "take care"] 
        self.help_requests = ["help", "assist", "support", "need help", "How does this work"]
        self.info_requests = ["what can you do", "features", "capabilities", "functions", "what do you do"]
        
        
        self.greeting_responses = [
            "Hello! I'm your virtual assistant. How can I help you today?",
            "Hi there! What can I do for you?",
            "Hey! How may I assist you?",
            "Greetings! How can I be of service? "
        ]
        self.farewell_responses = [
            "Goodbye! Have a great day!",
            "Bye! Take care!",
            "See you  later! Stay safe!",
            "Farewell! Hope to chat again soon!"
        ]
        self.help_responses = [
            "I'm here to assist you! You can ask me about my features or request specific help.",
            "How can I support you? Feel free to ask me about available commands.",
            "I'm here to help! Let me know what you need assistance with."
        ]
        self.info_responses = [
            "I can provide information, answer questions, and quide you through this virtual environment.",
            "My capabilities include answering querries, offering support, and assisting with navigation.",
            "I am designed to assist you with various tasks. How can I help you today?"
        ]
        
        
        
    # ---- INITIALIZING THE RESPOND METHOD ----
    # This function takes-in "user_input" as a String. and determines 
    # the appropriate response
    def respond(self, user_input):
        # .lower(): Converts the input to lowercase to make matching 
        # case-insensitive.
        # .strip(): Removes any extra spaces before or after the input.
        user_input = user_input.lower().strip()
        
        # If the user's input matches any greeting in self.greetings, 
        # the chatbot responds with a friendly greeting.
        if any(word in user_input for word in self.greetings):
            return random.choice(self.greeting_responses)
        # If the user's input matches any phrase in self.farewells, 
        # the chatbot says goodbye.
        elif any(word in user_input for word in self.farewells):
            return random.choice(self.farewell_responses)
        elif any(word in user_input for word in self.help_requests):
            return random.choice(self.help_responses)
        elif any(word in user_input for word in self.info_requests):
            return random.choice(self.info_responses)
        # If user_input is unrecognized
        else:
            return "I'm here to assist you. Can you please, clarify your request?"
        
# Example: usage
chatbot = VRChatbot()


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/chat', methods=['POST'])

def chat():
    user_message = request.json.get("message")
    bot_response = chatbot.respond(user_message)
    return jsonify({"response": bot_response})


if __name__ == '__main__':
    app.run(debug=True)
    
    
    
    
    
    
    
    
    
# Should greet the user with a random greeting response
# print(chatbot.respond("Hello"))
# Should say goodbye with a random farewell response
# print(chatbot.respond("end chat"))
# Should provide help responses
# print(chatbot.respond("help"))
# Should provide info responses
# print(chatbot.respond("what can you do"))