from flask import Flask, render_template, request, jsonify, make_response
from chatbot import init_components, generate_llm_response
import os
import pymongo
from dotenv import load_dotenv
import uuid
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "your-secret-key")

# MongoDB setup
mongo_client = pymongo.MongoClient(os.getenv("MONGO_URI"))
mongo_db = mongo_client["FAQ_chatbot"]
chat_history_collection = mongo_db["chat_history"]

# Initialize database with index
def init_database():
    try:
        chat_history_collection.create_index("session_id", unique=True)
        logger.info("Database index created successfully")
    except Exception as e:
        logger.error(f"Database initialization error: {str(e)}")

# Run database initialization
init_database()

def generate_session_id():
    """Generate a unique session ID."""
    return str(uuid.uuid4())

@app.route('/', methods=['GET', 'POST'])
def chatbot():
    # Get or create session_id from cookie
    session_id = request.cookies.get('session_id')
    if not session_id:
        session_id = generate_session_id()
        logger.info(f"New session created: {session_id}")

    # Initialize components with session_id
    vectordb, _, llm = init_components(session_id)

    if request.method == 'POST':
        try:
            user_message = request.form.get('user_message', '').strip()
            if not user_message:
                return jsonify({"error": "Please enter a message."}), 400

            # Fetch or initialize chat history
            history_doc = chat_history_collection.find_one({"session_id": session_id}) or {
                "session_id": session_id,
                "messages": [],
                "created_at": datetime.utcnow()
            }
            messages = history_doc.get("messages", [])

            # Add user message with timestamp
            user_message_entry = {
                "role": "user",
                "message": user_message,
                "timestamp": datetime.utcnow()
            }
            messages.append(user_message_entry)

            # Generate bot response
            bot_response = generate_llm_response(user_message, vectordb, llm, messages)
            bot_message_entry = {
                "role": "bot",
                "message": bot_response,
                "timestamp": datetime.utcnow()
            }
            messages.append(bot_message_entry)

            # Update chat history in database
            chat_history_collection.update_one(
                {"session_id": session_id},
                {
                    "$set": {
                        "messages": messages,
                        "last_updated": datetime.utcnow()
                    }
                },
                upsert=True
            )
            logger.info(f"Message saved for session {session_id}")

            response = jsonify({
                "bot_response": bot_response,
                "user_message": user_message,
                "timestamp": bot_message_entry["timestamp"].isoformat()
            })

            # Set session_id cookie if not already set
            if not request.cookies.get('session_id'):
                response.set_cookie('session_id', session_id, max_age=86400 * 30)  # 30 days

            return response

        except Exception as e:
            logger.error(f"Chatbot error: {str(e)}")
            return jsonify({"error": "An error occurred."}), 500

    # GET request: Render chat interface
    history_doc = chat_history_collection.find_one({"session_id": session_id}) or {"messages": []}
    response = make_response(render_template('chatbot.html', chat_history=history_doc.get("messages", [])))

    # Set session_id cookie if not already set
    if not request.cookies.get('session_id'):
        response.set_cookie('session_id', session_id, max_age=86400 * 30)  # 30 days

    return response

@app.route('/reset', methods=['POST'])
def reset_chat():
    try:
        # Get current session_id
        session_id = request.cookies.get('session_id')
        if not session_id:
            return jsonify({"error": "No active session."}), 400

        # Create a new session_id
        new_session_id = generate_session_id()
        logger.info(f"Resetting chat: Old session {session_id}, New session {new_session_id}")

        # Initialize new chat history
        chat_history_collection.update_one(
            {"session_id": new_session_id},
            {
                "$set": {
                    "messages": [],
                    "created_at": datetime.utcnow(),
                    "last_updated": datetime.utcnow()
                }
            },
            upsert=True
        )

        response = jsonify({"status": "success", "new_session_id": new_session_id})
        response.set_cookie('session_id', new_session_id, max_age=86400 * 30)  # 30 days
        return response

    except Exception as e:
        logger.error(f"Reset error: {str(e)}")
        return jsonify({"error": "Failed to reset chat."}), 500

if __name__ == '__main__':
    app.run(debug=True, port=7000, host='0.0.0.0')