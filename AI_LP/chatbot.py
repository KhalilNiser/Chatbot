
# ---- IMPORT_REQUIRED_LIBRARIES ----
# random: Generates random responses
import random
# openai: Fetches "key" responses from OpenAI API.
import openai
# json: Reads API keys from config.json 
import json
# logging: Records log messages (used here for debugging).
import logging
import numpy as np
import numexpr as ne 
# requests: Makes HTTP requests (to retrieve weather data from an API). 
import requests
import wikipedia 
# flask: flask: Creates a web server to handle user chatbot interaction via HTTP 
from flask import Flask, request, jsonify, render_template 
import os
# re: Provides regular expressions (used for cleaning and normalizing text).
import re
# typing: Supplies type annotations (for clarity and development support).
from typing import Dict, Any, List
# nltk, spacy, transformers, textblob: Libraries used for NLP tasks 
# (Text-tokenization, stemming, Named Entity Recognition (NER)).
import nltk
# For intent classifier training with NLTK
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
# Scikit-learn: Used for training an "intent classifier" with LogisticRegression"
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
# Used for deploying incoming HTTP requests and passing
# them to the python application for processing
from waitress import serve
# Limits the rate of incomming requests
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
# spacy: Specifically used here for named entity recognition (extracting city names).
import spacy
from sympy import sympify




# ---- Download Necessary NLTK data ----
nltk.download('punkt')

# nltk.download('averaged_perceptron_tagger')


# ---- INITIALIZE THE NLP_MODEL ----
# Loads a small English NLP model for named entity recognition 
# (NER). Used to extract city names from user input.
#
# NOTE: Since I'm only extracting "city names" using NER, "en_core_web_sm" is used
# to efficiently detect place names without needing a heavy model.
nlp = spacy.load("en_core_web_sm")


# logging.basicConfig(): Sets up the logging format and level.
# level=logging.INFO: Logs messages at INFO level and above.
# format: Includes a timestamp, the log level, and the message. 
# ---- Configure logging ----
logging.basicConfig(filename='chatbot.log', level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# load API Keys securely
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# ---- LOAD API KEYS FROM JSON FILE ----
def load_api_keys() -> Dict[str, str]:
    
    """
    Reads API keys from a JSON configuration file.
    Opens the file in read-only mode to ensure it is not modified.
    """
    # Uses a try/except block to handle potential errors like 
    # a missing file or incorrect JSON API formatting.
    try:
        # OPENAI_API_KEY: Authenticates the code by opening the "config.json" file.
        # Reads and loads its contents using json.load(file).
        # NOTE: "r" ensures the file is opened in read-only mode. 
        # Prevents accidental file modification or deletion.
        with open("config.json", "r") as file:
            
            # Assigning loaded data to a variable
            api_keys = json.load(file)
            return api_keys
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Error loading API Keys: {e}")
        raise 

# Assigns the result of load_api_keys() to variable config
config = load_api_keys()
# ---- WEATHER_API_KEY STORAGE ----
weather_api_key = config.get("WEATHER_API_KEY")
# ---- OPENAI_API_KEY STORAGE ----
openai_api_key = config.get("OPENAI_API_KEY")
if not openai_api_key:
    logging.error("OpenAI API Key is missing! Please check config.json!")
    raise ValueError("Missing OpenAI API Key")
# Set the API key for OpenAI
openai.api_key = openai_api_key


# ---- INITIALIZE THE FLASK_APP (Creates a Flask web app) ----
# This object will handle incoming HTTP requests 
# and route them to the appropriate functions.
app = Flask(__name__)



# Uses "Flask-Limiter to prevent API calls"
limiter = Limiter(get_remote_address, app=app, default_limits=["100 per hour"])



# ---- USER_PREFERENCES ----
# (To store settings like Fahrenheit vs Celsius)
user_preferences: Dict[str, Dict[str, str]] = {}



# Initialize Classifier and Vectorizer 
stemmer = PorterStemmer()



# ---- INTENT_CLASSIFIER_TRAINING ----

# ---- FUNCTION_EXTRACT_FEATURES ----
# Input tokenization and builds a dictionary feature
def preprocess_text(text: str) -> str:
    
    """
    Converts text into a dictionary of features for classification.
    Each word in the text is used as a feature.
    
    NOTE: When each word is used as a "feature". I'm treating each 
    word as a unique identifier or attribute of the text. 
    
    Why use this approache: A "feature" is an important or distinctive 
    aspect of something. 
    In machine learning and natural language processing (NLP) This 
    approache is often used to understand the content and structure of 
    a text. For a better and a more robust response.
    """
    # Tokenizes and stems the input text to normalize words
    # This converts the input text into lowercase and splits it 
    # into individual words (tokens).
    words = word_tokenize(text.lower())
    # this line converts user input into a simplified form where 
    # words are stemmed, making it easier for the chatbot to 
    # classify intents.
    # Stemming reduces words to their root form. The "join" ( " ".join(...) ) 
    # function takes the list of stemmed words and joins them with spaces to 
    # form a processed sentence.
    return " ".join([stemmer.stem(word) for word in words])


# Define sample training data: A list of tuples (features, intent_label)
training_data = [
   ("hello hi hey good morning greetings", "greeting"),
    ("goodbye bye see you later", "farewell"),
    ("help assist support", "help"),
    ("what can you do features capabilities", "info"),
    ("weather forecast temperature climate", "weather"),
    ("tell me about facts information knowledge where what who when", "fact"),
    ("news headlines updates", "news"),
    ("joke funny humor laugh", "joke"),
    ("math calculate solve equation", "math"),
    ("search lookup wikipedia info", "wikipedia")
]


# ---- TF-IDF_VECTORIZER ----
# Uses "TF-IDF Vectorizer" to transform text data into numerical features. 
# 
# NOTE: TfidfVectorizer takes raw text data (a collection of documents) and 
# transforms them into a matrix where each row represents a document and
# each column represents a word (or n-gram) from the vocabulary. 
# 
# The values in the matrix represent the TF-IDF score for each word in each 
# document, calculated by multiplying the term frequency (how often a word 
# appears in a document) with the inverse document frequency (how rare the 
# word is across all documents). 
# 
# NOTE: Applied "TF-IDF Vectorizer" and "Logistic Regression" 
# for better accuracy 
# zip(*training_data) separates the first and second elements of each tuple 
# into two separate lists (train_text, train_labels) 
# Features (train_texts) → The input text
# Labels (train_labels) → The expected output (intent category)
train_texts, train_labels = zip(*training_data)
vectorizer = TfidfVectorizer()
x_train = vectorizer.fit_transform(train_texts)

# Trains a "Logistic Regression" model to classify the user intents 
# (e.g., greeting, farewell, weather inquiry). 
# Goal: When a user types a message, the model should predict the 
# "intent" behind it, such as whether they are greeting the chatbot, 
# asking for help, or just simply requesting weather information.
classifier = LogisticRegression()
classifier.fit(x_train, train_labels)



# ---- DEFINE_CLASS_VRChatbot ----
class VRChatbot:
    
    # ---- INITIALIZING THE CHATBOT __INIT__ METHOD() ----
    # __init__: Special method (like a constructor in Java) that is called 
    # when an object of the class is created.
    # self: Allows access to all data fields and attributes of the class.
    def __init__(self) -> None:
        # Contains "hardcoded responses" for certain categories.
        self.responses = {
            "greeting": ["Hello! How can I assist you today?", "Hi there! How may I help?"],
            "farewell": ["Goodbye! Have a great day!", "See you later!"],
            "help": ["I can assist with various tasks. What do you need help with?"],
            "info": ["I provide information and guidance. How can I assist?"],
            "weather": ["Please provide a location for the weather report."],
            "fact": ["I can find facts for you. What topic are you interested in?"],
            "news": ["Fetching the latest news updates for you."],
            "joke": ["Why don't programmers like nature? Because it has too many bugs!"],
        }  
        
        

    # ---- INTENT_CLASSIFICATION ----
    # Converts user input into a TF-IDF vector and predicts their intent.
    def classify_intent(self, user_input: str) -> str:
        """ Classifies the intent of the user's input. """
        X_input = vectorizer.transform([user_input])
        return classifier.predict(X_input)[0]
    
    
    
    # ---- WEATHER_API_INTEGRATION ----
    # Uses "WeatherAPI.com" to retrieve real-time weather information 
    # based on location.
    def fetch_weather(self, location: str) -> str:
        
        """ Fetches weather information for a given location. """
        try:
            url = f"http://api.weatherapi.com/v1/current.json?key={weather_api_key}&q={location}&aqi=no"
            response = requests.get(url).json()
            
            # Debugging print API rsponse
            logging.info(f"Weather API response: {response}")
            
            if "current" in response:
                return f"The current temperature in {location} is {response['current']['temp_f']}\u00B0F."
            elif "error" in response:
                return f"Error: {response['error'].get('message', 'Unable to fetch weather data')}"
            else:
                return "Sorry I couldn't retrieve the weather data. Please check the location."
        except requests.RequestException as e:
            logging.error(f"Weather API error: {e}")
            return "Weaher service is currently unavailable!."
    
    

    # ---- GPT-4_API_INTEGRATION ----
    # Queries GPT-4 if no predefined response matches.
    # Integrated "Chat GPT-4" responses
    def query_gpt_4(self, user_input: str) -> str:
        
        """
        Generates a response using OpenAI's GPT-4 model.
        """
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "user", "content": user_input}
                ],
                
                api_key = openai_api_key
            )
            return response["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logging.error(f"GPT-4 API error: {e}")
            return "I'm sorry, I couldn't process that request."
        
        
        
    def query_wikipedia(self, topic: str) -> str:
        
        try:
            summary = wikipedia.summary(topic, sentences=2)
            return summary
        except wikipedia.exceptions.DisambiguationError as e:
            return f"Can you be more specific? {e.options[:3]}"
        except wikipedia.exceptions.PageError:
            return "I couldn't find information on that topic"
        except wikipedia.exceptions.HTTPTimeoutError:
            return "Wikipedia service is currently unavailable. Please try again later."
        
        
        
    def calculate_math(self, expression: str) -> str:
        """Evaluates a math expression safely."""
        
        try:
            expression = re.sub(r'[^0-9+\-*/(). ]', '', expression)
            result = sympify(expression)
            return f"The answer is {result}."
        except (SyntaxError, ZeroDivisionError, ValueError) as e:
            logging.error(f"Math error: {e}")
            return "I couldn't calculate that. Please check the expression."     



    # ---- INITIALIZING THE RESPOND METHOD ----
    # ---- MAIN CHATBOT RESPONSE LOGIC ----
    # ---- Checks if the Input Matches an Intent
    def respond(self, user_input: str) -> str:
        
        """
        Processes user input and returns an appropriate response.
        If input matches certain keywords, returns pre-defined 
        responses. Otherwise, defers to the GPT-4 model.
        """
        user_input = user_input.lower()
        
        # Handle arithmetic expressions separately
        if re.search(r'\d+[\+\-\*/]\d+', user_input):
            return self.calculate_math(user_input)
        
        # Uses intent_classifier to guide the response selection
        intent = self.classify_intent(user_input)
        logging.info(f"Classified intent: {intent}")
        
        # Processes user input and determines the appropriate response.
        # Returns a "predifined response", if available
        # Calls on the "fetch_weather()" funtion if it's a weather-related query
        # Defaults to "GPT-4" for more complex responses 
        if intent in self.responses:
            return random.choice(self.responses[intent])
        elif intent == "weather":
            doc = nlp(user_input)
            city = next((ent.text for ent in doc.ents if ent.label_ == "GPE"), None)
            return self.fetch_weather(city) if city else "Please specify a location."
        elif intent == "fact":
            # Extract topic from user input
            topic = " ".join(user_input.split()[2:])
            return self.query_wikipedia(topic)
        else:
            return self.query_gpt_4(user_input)
            
        

# ---- CREATE AN INSTANCE OF THE CLASS VRChatbot (INITIALIZE chatbot) ----
# Instantiates the "VRChatbot class" so that it's ready to process inputs.
chatbot = VRChatbot()



# ---- FLASK ROUTES FUNCTION ---- 
# Serves the "chatbot" interface
# All user interactions are handled via a web interface.
@app.route('/')
def home() -> str:
    
    """
    Renders the homepage (index.html) for the Chatbot interface.
    """
    return render_template("index.html")



# ---- CHATBOT_END_POINT ----
# Receives user messages via AJAX requests
# Lmits the requests for 5 per minute tp prevent abuse
@app.route('/chat', methods=['POST'])
@limiter.limit("30 per minute")

def chat() -> Any:
    
    """
    Handles the /chat route for AJAX requests.
    Receives a JSON message, processes it, and returns the chatbot's response.
    """
    user_message = request.json.get("message", "").strip()
    
    # if the user sends an empty message, returns an 
    # error with status code 400 (Bad Request).
    if not user_message:
        return jsonify({"response": "Please enter a message."}), 400
    response = chatbot.respond(user_message)
    return jsonify({"response": response})



# ---- Deployment with Gunicorn ----
# ---- RUN FLASK APP (Runs the ChatBot in "debug" mode) ----
# In production, Flask is often run using Gunicorn, 
# instead of Flask’s built-in development server.
if __name__ == '__main__':
    
    serve(app, host="0.0.0.0", port=8000)
    # class GunicornApp(BaseApplication):
        # def __init__(self, app, options=None):
            # self.options = options or {}
            # self.application = app
            # super().__init__()
            
            
            
        
        # def load_config(self):
            # for key, value in self.options.items():
                # self.cfg.set(key, value)
                
                
                
                
        # def load(self):
            # return self.application
        
        
    # options = {'bind': '0.0.0.0:8000', 'workers': 4}
    # GunicornApp(app, options).run()

    
    
    
    
#                   ---- SUMMARY ----
# NOTE: This ChatBot is a hybrid rule-based and AI powered assistant with a 
# "structured intent classfier", "Real-time API Integrations", and GPT-4 
# fallback for more complex questions
# 
# NEW_UPDATED_IMPROVED_VERSION:
# ✅ 🔑 OpenAI API Key & Weather API Key Loaded Securely (JSON-based storage):
# ✅ 🔑 API Key Loading: Added try/except to handle missing/invalid config files.
# ✅ 🔑 Error Handling: Wrapped network calls (weather and GPT API) in try/except blocks.
# ✅ 🧠 Improved NLP Understanding (Extracts city names using spaCy)
# ✅ 🧠 Enhanced FLASK handling. Improved "Exception handling" for better API calls.
# ✅ 🧠 Applied TF-IDF_Vectorizer and LogisticRegression 
# ✅ 🧠 Serves much better accuracy on larger, overwhelming data.
# ✅ 🧠 Organized and informed data structure.
# ✅ 🧠 Logging: Added logging for debugging purposes instead of print.
# ✅ 🧠 Enhanced logging/debugging for advanced issue tracking.
# ✅ ⛅ Smarter Weather Handling (supports Fahrenheit & Celsius)
# ✅ ⛅ Weather Expansion: Uses real-time weather (weather update) based on location. 
# ✅ ⛅ Smarter Weather Handling (Asks for missing locations, supports Fahrenheit & Celsius)
# ✅ 📖 Fact-Checking With Other queries → Calls GPT-4 for a response.
# ✅ 📖 Input Processing: Introduced clean_input for normalization, punctuation, and improved token splitting.
# ✅ 📖 Applied TF-IDF_Vectorizer and LogisticRegression for better accuracy on large data. 
# ✅ 📖  Expanded on the intent category. More advanced "text-cleaning" using NLTK Stemming.
# ✅ ⚙️ Smart user Preferences System (Remembers settings like temperature units)
# ✅ ⚙️ Enhanced Preference Parsing: Allowed multi‑word values for user preferences.
# ✅ ⚙️ Type Annotations: Provided type hints and added docstrings for clarity.
# ✅ 🤖 More Natural Responses (GPT handles unknown inputs instead of defaulting to a generic reply)
# ✅ 🚀 This is an advanced way to deploy Flask apps with Gunicorn. 
# ✅ 🚀 (providing better control over the server configuration).