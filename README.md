# **** Chatbot Project ****

## Introduction 

This project is a smart, interactive **AI-powered chatbot** designed to simulate real-time conversations using a 
blend of rule-based logic and machine learning models. This chatbot is capable of handling greetings, farewells, 
help requests, and basic information queries. It also includes NLP-based understanding to analyze user input and 
generate relevant responses. The motivation behind this project is to create a foundational intelligent assitant 
to could be expanded into educational, customer service, or VR/AR-based environments.

***

## Getting Started

This section helps you set up the chatbot on your local machine.

### Installation Process

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/chatbot-project.git
   cd chatbot-project
    ```

2. (Optional) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Mac/Linux
   venv\Scripts\activate      # Windows
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Software Dependencies

The chatbot uses the following Python libraries:
NOTE: nltk, spacy, transformers, textblob: Libraries used for NLP tasks 
(Text-tokenization, stemming, Named Entity Recognition (NER)).
- `nltk` – Natural Language Toolkit for tokenization and basic NLP tasks
- `word_tokenize, PorterStemmer` – For named entity recognition and dependency parsing (For intent classifier training with NLTK)
- `Scikit-learn, TfidfVectorizer, LogisticRegression` – Used for training an "intent classifier" with LogisticRegression"
- `waitress` – # Used for deploying incoming HTTP requests and passing them to the python application for processing
- `limiter` – Limits the rate of incomming requests
- `numexpr` – For safe evaluation of user-input expressions
- `openai` – Fetches "key" responses from OpenAI API.
- `json` – Reads API keys from config.json 
- `logging` – Records log messages (used here for debugging).
- `requests` – Makes HTTP requests (to retrieve weather data from an API). 
- `re` – Provides regular expressions (used for cleaning and normalizing text).
- `flask` – For creating a lightweight web interface (optional)
- `numexpr` – For safe evaluation of user-input expressions

### Latest Releases

- **Version 1.0.0**: Basic chatbot functionality implemented with keyword and pattern matching.
- **Version 1.1.0** *(planned)*: Improved NLP, more robust conversation memory, and adaptive responses.

### API References

If you are using a web interface:
- Endpoint: `POST /chat`
- Request body:
  ```json
  {
    "message": "Hello, chatbot!"
  }
  ```
- Response body:
  ```json
  {
    "response": "Hi there! How can I assist you today?"
  }
  ```

---
# Build and Test

### Building the Code

No build step is required. Just run the main Python script:

```bash
python chatbot.py
```

Or, if you're using the Flask web app:
```bash
python app.py
```

### Running Tests

Basic unit tests are included. To run tests:

```bash
python -m unittest discover tests/
```

> You can add more tests to ensure new features work correctly without breaking existing functionality.

---
# Contribute

I welcome contributions! Here's how you can help:

1. Fork this repo and clone your copy.
2. Create a new branch for your feature or bug fix:
   ```bash
   git checkout -b feature-branch
   ```
3. Make your changes and test thoroughly.
4. Commit and push your code.
5. Submit a Pull Request with a clear explanation of what you added or fixed. 

### Contributor Guidelines

- Use clear and concise commit messages.
- Ensure code adheres to PEP 8 styling.
- Test thoroughly before submitting.
- Document any new functions or modules you add.
