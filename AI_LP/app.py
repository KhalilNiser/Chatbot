# -*- coding: utf-8 -*- 

# ---- IMPORT_REQUIRED_LIBRARIES ----
from flask import Flask, request, jsonify, render_template 
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from waitress import serve
from redis import Redis
import openai
# IMPORT chatbot from ChatBot folder
from chatbot import VRChatbot


app = Flask(__name__)


# Connect to Redis for rate limiting storage
redis_client = Redis(host="localhost", port=6379, db=0)

limiter = Limiter(
    get_remote_address, 
    app=app, 
    storage_uri = "redis://localhost:6379"
)


# ---- CREATE AN INSTANCE OF THE CLASS VRChatbot (INITIALIZE chatbot) ----
# Instantiates the "VRChatbot" class: Allowing access to work with 
# data, attributes, and functionalities defined within this class
chatbot = VRChatbot()

# ---- FLASK ROUTES FUNCTION ---- 
# Serves the "chatbot" interface
# All user interactions are handled via a web interface.
@app.route('/')
def home():
    
    """
    Renders the homepage (index.html) for the chatbot interface
    """
    return render_template("index.html")


@app.route('/chat', methods=['POST'])
@limiter.limit("5 per minute")



def chat():
    
    """
    Handles the /chat route for AJAX requests.
    Receives a JSON message, processes it, and returns the chatbot's response.
    """
    user_message = request.json.get("message", "").strip()
    
    # if the user sends an empty message, returns an 
    # error with status code 400 (Bad Request).
    if not user_message:
        return jsonify({"response": "Please enter a message."}), 400
    
    try:
        bot_response = chatbot.respond(user_message)
        # ✅ Properly returns response on success
        return jsonify({"response": bot_response})  
    except Exception as e:   
        return jsonify({"response": "An error occurred. Please try again later."}), 500


# ---- RUN FLASK APP (Runs the ChatBot in "debug" mode) ----
if __name__ == '__main__':
    serve(app, host="0.0.0.0", port=5000)
    
    
    
    
    
    
    
    
    
# Should greet the user with a random greeting response
# print(chatbot.respond("Hello"))
# Should say goodbye with a random farewell response
# print(chatbot.respond("end chat"))
# Should provide help responses
# print(chatbot.respond("help"))
# Should provide info responses
# print(chatbot.respond("what can you do"))

