from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
from groq import Groq
import re

# Load environment variables from .env file
load_dotenv()

# Initialize Groq client with your API key
groq_api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=groq_api_key)

# Initialize Flask app
app = Flask(__name__)

# Global variable to track chat history (In a production system, you should use a session store like Redis)
chat_history = []


# Function to simulate room search (you can replace this with a real search function)
def search_rooms(location, budget):
    # Example room data (you can replace this with dynamic data from an API or database)
    rooms = [{
        "Hotel_name": "Oyo Hotel Central",
        "Location": location,
        "Budget": 3000,
        "Amenities": "Free Wi-Fi, Breakfast",
        "Rating": 4.5,
        "Discount": "10%"
    }, {
        "Hotel_name": "Budget Stay Inn",
        "Location": location,
        "Budget": 2500,
        "Amenities": "Wi-Fi, Parking",
        "Rating": 4.2,
        "Discount": "15%"
    }, {
        "Hotel_name": "Luxury Suite Palace",
        "Location": location,
        "Budget": 4000,
        "Amenities": "Pool, Gym, Free Breakfast",
        "Rating": 4.8,
        "Discount": "5%"
    }]
    return [room for room in rooms if budget[0] <= room['Budget'] <= budget[1]]


def get_answer(chat_history, user_query, user_location, user_budget):
    room_options = ""

    if user_location and user_budget:
        matching_rooms = search_rooms(user_location, user_budget)
        if matching_rooms:
            # Format the room details with the required information
            rooms_list = "\n".join([
                re.sub(
                    r'[*_]',
                    '',  # Remove unwanted characters
                    f"Room {index + 1}: {room['Hotel_name']} in {room['Location']}, "
                    f"₹{room['Budget']} per night (Amenities: {room['Amenities']}, "
                    f"Rating: {room['Rating']}⭐, Discount: {room['Discount']})"
                ) for index, room in enumerate(matching_rooms)
            ])
            room_options = f"Here are some hotel options for {user_location} within your budget:\n\n{rooms_list}\n\nPlease select one of the available rooms by typing '1' for Room 1, '2' for Room 2, or '3' for Room 3."
        else:
            room_options = f"Sorry, we couldn't find any hotel options in {user_location} within your budget."
    prompt_template = f"""
    You are an Oyo bot that assists with hotel booking by asking for the location, check-in/check-out dates, number of guests, and budget. After gathering the information, generate room options and provide a reference number. Avoid using emojis or special characters.

    ### Chat History:
    {chat_history}

    ### Your question:
    {user_query}
"""



    try:
        completion = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{
                "role": "system",
                "content": prompt_template
            }],
            temperature=0.5,  # Lower temperature for more deterministic responses
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

    # Store user preferences
    user_location = req_data.get('queryResult', {}).get('parameters',
                                                        {}).get('location')
    user_budget = req_data.get('queryResult', {}).get('parameters', {}).get(
        'budget', (0, 10000))  # Example budget range

    # Append the current query to the chat history
    chat_history.append({"role": "user", "content": query_text})

    # Handle the 'input.unknown' action
    if action == 'input.unknown':
        response = get_answer(chat_history, query_text, user_location,
                              user_budget)

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
