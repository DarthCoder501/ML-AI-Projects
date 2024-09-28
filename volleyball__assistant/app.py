import os
from flask import render_template, request, jsonify
from flask import Flask
from groq import Groq
from models import db, Player

# Create an instance of the Flask application
app = Flask(__name__)

# Retrieve the API key from environment variables
api_key = os.environ.get("GROQ_API_KEY")
print(f"GROQ_API_KEY: {api_key}")  # Debug print to check if the API key is loaded

if not api_key:
    raise ValueError("GROQ_API_KEY environment variable not set.")

# Initialize the Groq client
client = Groq(api_key=api_key)

# Root route
@app.route('/')
def home():
    return render_template('training_plan.html')

@app.route('/api/get_training_plan', methods=['POST'])
def get_training_plan():
    data = request.json
    player_name = data['name']
    player_goals = data['goals']
    
    # Validate input data
    if 'name' not in data or 'goals' not in data:
        return jsonify({'error': 'Missing player name or goals.'}), 400

    # Save player info to the database
    new_player = Player(name=player_name, goals=player_goals)
    db.session.add(new_player)
    db.session.commit()

    # Prepare the chat message for the Groq API request
    messages = [
        {
            "role": "user",
            "content": f"Create a personalized training plan for a volleyball player named {player_name} with the following goals: {player_goals}."
        }
    ]

    try:
        # Make the chat completion request
        chat_completion = client.chat.completions.create(
            messages=messages,
            model="llama3-8b-8192",  # Ensure this model is available
        )

        # Extract the training plan from the response
        training_plan = chat_completion.choices[0].message.content

        # Example of additional tool use to generate drills
        drill_messages = [
            {
                "role": "user",
                "content": f"Provide some specific drills to help {player_name} achieve their goals: {player_goals}."
            }
        ]

        drill_completion = client.chat.completions.create(
            messages=drill_messages,
            model="llama3-8b-8192",  # Reusing the same model for consistency
        )

        drills = drill_completion.choices[0].message.content

    except Exception as e:
        # Log the error for debugging
        print(f"Request failed: {e}")
        return jsonify({'error': 'Failed to retrieve training plan. Please try again later.'}), 500

    return jsonify({
        'plan': training_plan,
        'drills': drills
    })

if __name__ == '__main__':
    app.run(debug=True)
