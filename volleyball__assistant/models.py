# models.py
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Player(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    goals = db.Column(db.Text, nullable=False)

    def __repr__(self):
        return f'<Player {self.name}>'
