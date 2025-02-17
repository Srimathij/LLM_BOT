from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
import re
import requests
from bs4 import BeautifulSoup
from groq import Groq

# Load environment variables from .env file
load_dotenv()

# Initialize Groq client with your API key
groq_api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=groq_api_key)

# Initialize Flask app
app = Flask(__name__)

# Global variable to track chat history (In a production system, you should use a session store like Redis)
chat_history = []


def get_answer(chat_history, user_query):
    # Scrape the website for relevant content based on the user query
    scraped_content = scrape_website(user_query)

    prompt_template = f"""
    You are a bot that assists with information from the Sompo website. You should answer questions based on the content available on the website.
    https://www.sompo.com.sg/ - website link you should get response from this

    ### Chat History:
    {chat_history}

    ### User Question:
    {user_query}

    ### Website Response:
    {scraped_content}
    """

    try:
        completion = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{
                "role": "system",
                "content": prompt_template
            }],
            temperature=
            0.5,  # Lower temperature for more deterministic responses
            max_tokens=500,  # Limit tokens for shorter responses
            top_p=0.9,
            stream=False,
        )

        response_text = completion.choices[0].message.content.strip()
        return response_text

    except Exception as e:
        print("Error occurred:", e)
        return "An error occurred while processing your request."


@app.route('/', methods=['GET'])
def home():
    return "Server is running", 200


@app.route('/dialogflow', methods=['POST'])
def dialogflow():
    global chat_history

    # Get the JSON data from Dialogflow
    req_data = request.get_json()

    # Print the request data for debugging purposes
    print(req_data)

    # Extract action and query text from the request
    action = req_data.get('queryResult', {}).get('action')
    query_text = req_data.get('queryResult', {}).get('queryText')

    # Append the current query to the chat history
    chat_history.append({"role": "user", "content": query_text})

    # Handle the 'input.unknown' action
    if action == 'input.unknown':
        response = get_answer(chat_history, query_text)

        # Append the system response to the chat history
        chat_history.append({"role": "system", "content": response})

        return jsonify({'fulfillmentText': response})
    else:
        # Handle unknown actions
        return jsonify(
            {'fulfillmentText': f'No handler for the action {action}.'})


# Main function to run the Flask app
if __name__ == '__main__':
    port = int(os.getenv('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
