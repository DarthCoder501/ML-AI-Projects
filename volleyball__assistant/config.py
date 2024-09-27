# config.py
import os

class Config:
    SQLALCHEMY_DATABASE_URI = 'sqlite:///volleyball_assistant.db'  # Use SQLite for simplicity
    SQLALCHEMY_TRACK_MODIFICATIONS = False
