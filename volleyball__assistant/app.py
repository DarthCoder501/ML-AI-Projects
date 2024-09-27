# app.py
from flask import render_template, request, jsonify
from flask import Flask
import flask
import requests
from models import db, Player

# Create an instance of the Flask application
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('training_plan.html')

@app.route('/api/get_training_plan', methods=['POST'])
def get_training_plan():
    data = request.json
    player_name = data['name']
    player_goals = data['goals']

    # Save player info to the database
    new_player = Player(name=player_name, goals=player_goals)
    db.session.add(new_player)
    db.session.commit()

    # Call the GROQ API to get the personalized training plan
    groq_response = requests.post(
        'https://groq-api-endpoint.com',  # Replace with actual endpoint
        json={
            'player_name': player_name,
            'goals': player_goals
        }
    )
    training_plan = groq_response.json().get('plan', 'No plan available.')

    return jsonify({'plan': training_plan})

if __name__ == '__main__':
    app.run(debug=True)
