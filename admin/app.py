from flask import Flask, jsonify, render_template, request, redirect, url_for, session, send_file
from pymongo import MongoClient
from functools import wraps
from datetime import datetime
import os
import logging
import bcrypt
from dotenv import load_dotenv
import io
import csv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your-secret-key')

# MongoDB setup
mongo_client = MongoClient(os.getenv('MONGO_URI'))
db = mongo_client['FAQ_chatbot']
chat_history_collection = db['chat_history']

# Admin credentials (hardcoded for single admin; use env vars in production)
ADMIN_USERNAME = 'admin'
ADMIN_PASSWORD_HASH = bcrypt.hashpw('admin123'.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

# Login required decorator


# Route: Admin dashboard


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)