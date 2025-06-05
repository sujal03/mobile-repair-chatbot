from flask import (
    Flask, render_template, request, jsonify, make_response, redirect, url_for, send_file, session
)
from utils.chatbot import (
    generate_llm_response, generate_suggestions, chat_history_collection
)
import io
from functools import wraps
import csv
import os
import bcrypt
from dotenv import load_dotenv
import uuid
from datetime import datetime
import logging
from io import BytesIO
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("chatbot.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "your-secret-key")
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # Limit uploads to 10MB

ADMIN_USERNAME = os.getenv("ADMIN_USERNAME")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")
ADMIN_PASSWORD_HASH = bcrypt.hashpw(ADMIN_PASSWORD.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def generate_session_id():
    """Generate a unique session ID using UUID.

    Returns:
        str: A unique session identifier.
    """
    return str(uuid.uuid4())


def get_or_create_session():
    """Get the existing session ID from cookies or create a new one.

    Returns:
        str: The session ID.
    """
    session_id = request.cookies.get('session_id')
    if not session_id or not is_valid_session(session_id):
        session_id = generate_session_id()
        logger.info(f"New session created: {session_id}")
    else:
        logger.info(f"Using existing session: {session_id}")
    return session_id


def is_valid_session(session_id):
    """Check if a session exists and is valid in MongoDB.

    Args:
        session_id (str): The session ID to validate.

    Returns:
        bool: True if valid, False otherwise.
    """
    try:
        session = chat_history_collection.find_one({"session_id": session_id})
        return session is not None
    except Exception as e:
        logger.error(f"Error checking session validity: {str(e)}")
        return False

def get_or_create_session(force_new=False):
    """Get the existing session ID from cookies or create a new one.

    Args:
        force_new (bool): If True, always create a new session ID, ignoring existing cookies.

    Returns:
        str: The session ID.
    """
    session_id = request.cookies.get('session_id')
    if force_new or not session_id or not is_valid_session(session_id):
        session_id = generate_session_id()
        logger.info(f"New session created: {session_id}")
    else:
        logger.info(f"Using existing session: {session_id}")
    return session_id

# Route: Redirect to chatbot
@app.route('/', methods=['GET'])
def index():
    """Redirect to the chatbot interface."""
    return redirect(url_for('chatbot'))


# Route: Chatbot interface
@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    """Handle GET and POST requests for the chatbot interface.

    GET: Render the chatbot template with chat history (force new session on page load).
    POST: Process user input (text or image), generate a response, and update history.

    Returns:
        Response: Rendered template (GET) or JSON response (POST).
    """
    if request.method == 'GET':
        # Force a new session on page load (e.g., page reload or first visit)
        session_id = get_or_create_session(force_new=True)
        
        # Since we reset the session on page load, chat history should be empty
        history_doc = chat_history_collection.find_one({"session_id": session_id})
        chat_history = history_doc.get("messages", []) if history_doc else []
        
        response = make_response(render_template('index.html', chat_history=chat_history))
        response.set_cookie('session_id', session_id, max_age=86400 * 30)  # 30 days
        return response

    elif request.method == 'POST':
        try:
            session_id = request.cookies.get('session_id') or generate_session_id()
            user_message = request.form.get('user_message', '').strip()
            image_file = request.files.get('image')

            # Validate input
            if not user_message and not image_file:
                return jsonify({"error": "Please provide a message or image."}), 400

            if image_file:
                if not image_file.content_type.startswith('image/'):
                    return jsonify({"error": "Invalid file type. Upload an image."}), 400
                if image_file.content_length > 5 * 1024 * 1024:
                    return jsonify({"error": "Image exceeds 5MB limit."}), 400

            # Retrieve or initialize chat history
            history_doc = chat_history_collection.find_one({"session_id": session_id})
            messages = history_doc.get("messages", []) if history_doc else []

            # Save user message
            user_entry = {
                "role": "user",
                "message": user_message or "Image uploaded for analysis",
                "timestamp": datetime.utcnow()
            }
            if image_file:
                import base64
                image_data = BytesIO(image_file.read())
                image_base64 = base64.b64encode(image_data.getvalue()).decode('utf-8')
                user_entry["image_base64"] = image_base64
                user_entry["image_url"] = f"data:image/jpeg;base64,{image_base64}"

            messages.append(user_entry)
            chat_history_collection.update_one(
                {"session_id": session_id},
                {"$push": {"messages": user_entry}, "$set": {"last_updated": datetime.utcnow()}},
                upsert=True
            )
            logger.info(f"User message saved for session {session_id}")

            # Generate and save bot response
            start_time = time.time()
            bot_response = generate_llm_response(
                user_message, messages,
                image_file=BytesIO(image_data.getvalue()) if image_file else None
            )
            logger.info(f"Response generated in {time.time() - start_time:.2f} seconds")

            bot_entry = {"role": "bot", "message": bot_response, "timestamp": datetime.utcnow()}
            messages.append(bot_entry)
            chat_history_collection.update_one(
                {"session_id": session_id},
                {"$push": {"messages": bot_entry}, "$set": {"last_updated": datetime.utcnow()}}
            )
            logger.info(f"Bot response saved for session {session_id}")

            # Generate suggestions
            suggestions = generate_suggestions(user_message, messages)

            # Prepare response
            response_data = {
                "bot_response": bot_response,
                "user_message": user_message,
                "timestamp": bot_entry["timestamp"].isoformat(),
                "suggestions": suggestions
            }
            response = jsonify(response_data)
            response.set_cookie('session_id', session_id, max_age=86400 * 30)
            return response

        except Exception as e:
            logger.error(f"Error processing request: {str(e)}", exc_info=True)
            return jsonify({"error": "Processing error.", "details": str(e) if app.debug else "Try again."}), 500


# Route: Get suggestions
@app.route('/get_suggestions', methods=['POST'])
def get_suggestions():
    """Generate follow-up question suggestions based on user input and history.

    Returns:
        JSON: List of suggested questions.
    """
    try:
        data = request.get_json() or {}
        user_message = data.get('user_message', '').strip()
        session_id = request.cookies.get('session_id')
        messages = []
        if session_id:
            history_doc = chat_history_collection.find_one({"session_id": session_id})
            messages = history_doc.get("messages", []) if history_doc else []
        suggestions = generate_suggestions(user_message, messages)
        return jsonify({"suggestions": suggestions})
    except Exception as e:
        logger.error(f"Error generating suggestions: {str(e)}")
        return jsonify({
            "error": "Failed to generate suggestions.",
            "suggestions": ["How to fix my screen?", "Battery drain tips?", "Reset my device?"]
        }), 500


# Route: Reset chat
@app.route('/reset', methods=['POST'])
def reset_chat():
    """Reset the chat by creating a new session.

    Returns:
        JSON: Status and new session ID.
    """
    try:
        session_id = request.cookies.get('session_id')
        if not session_id:
            return jsonify({"error": "No active session."}), 400
        new_session_id = generate_session_id()
        logger.info(f"Resetting chat: Old {session_id}, New {new_session_id}")
        chat_history_collection.update_one(
            {"session_id": new_session_id},
            {"$set": {"messages": [], "created_at": datetime.utcnow(), "last_updated": datetime.utcnow()}},
            upsert=True
        )
        response = jsonify({"status": "success", "new_session_id": new_session_id})
        response.set_cookie('session_id', new_session_id, max_age=86400 * 30)
        return response
    except Exception as e:
        logger.error(f"Reset error: {str(e)}")
        return jsonify({"error": "Failed to reset chat."}), 500
    

# Route: Admin dashboard
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('admin_logged_in'):
            return redirect(url_for('admin_login', next=request.url))
        return f(*args, **kwargs)
    return decorated

# Admin login page
@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password').encode('utf-8')
        if username == ADMIN_USERNAME and bcrypt.checkpw(password, ADMIN_PASSWORD_HASH.encode('utf-8')):
            session['admin_logged_in'] = True
            next_url = request.form.get('next') or url_for('admin_panel')
            return redirect(next_url)
        else:
            return render_template('admin_login.html', error='Invalid credentials', next=request.form.get('next'))
    return render_template('admin_login.html', next=request.args.get('next'))

# Admin logout
@app.route('/admin/logout')
def admin_logout():
    session.pop('admin_logged_in', None)
    return redirect(url_for('admin_login'))

# Admin panel
@app.route('/admin')
@login_required
def admin_panel():
    return render_template('admin.html')

# API to get all conversations
@app.route('/api/admin/conversations', methods=['GET'])
@login_required
def get_conversations():
    try:
        pipeline = [
            {'$match': {'session_id': {'$exists': True}, 'messages': {'$exists': True, '$ne': []}}},
            {'$unwind': '$messages'},
            {'$sort': {'messages.timestamp': -1}},
            {'$group': {
                '_id': '$session_id',
                'latest_message': {'$first': {'$ifNull': ['$messages.message', '']}},
                'latest_role': {'$first': {'$ifNull': ['$messages.role', 'unknown']}},
                'latest_timestamp': {'$first': {'$ifNull': ['$messages.timestamp', datetime.min]}},
                'message_count': {'$sum': 1}
            }},
            {'$sort': {'latest_timestamp': -1}}
        ]
        conversations = list(chat_history_collection.aggregate(pipeline))
        total_conversations = chat_history_collection.count_documents({'session_id': {'$exists': True}, 'messages': {'$exists': True, '$ne': []}})
        logger.info(f"Found {len(conversations)} conversations")
        return jsonify({
            'status': 'success',
            'conversations': [
                {
                    'session_id': str(conv['_id']),
                    'latest_message': conv['latest_message'][:100] + ('...' if len(conv['latest_message']) > 100 else ''),
                    'latest_role': conv['latest_role'],
                    'latest_timestamp': conv['latest_timestamp'].isoformat(),
                    'message_count': conv['message_count']
                } for conv in conversations
            ],
            'total_conversations': total_conversations
        })
    except Exception as e:
        logger.error(f"Error fetching conversations: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

# API to get conversation by session_id
@app.route('/api/admin/conversation/<session_id>', methods=['GET'])
@login_required
def get_conversation(session_id):
    try:
        conversation = chat_history_collection.find_one({'session_id': session_id})
        if not conversation or not conversation.get('messages'):
            logger.warning(f"No conversation found for session_id: {session_id}")
            return jsonify({'status': 'error', 'message': 'Conversation not found'}), 404
        messages = conversation['messages']
        logger.info(f"Found {len(messages)} messages for session_id: {session_id}")
        return jsonify({
            'status': 'success',
            'messages': [
                {
                    'id': str(msg.get('id', '')) or f"{session_id}_{idx}",
                    'role': msg.get('role', 'unknown'),
                    'message': msg.get('message', ''),
                    'timestamp': msg.get('timestamp', datetime.min).isoformat()
                } for idx, msg in enumerate(messages)
            ]
        })
    except Exception as e:
        logger.error(f"Error fetching conversation {session_id}: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

# API to delete conversations
@app.route('/api/admin/conversations/delete', methods=['POST'])
@login_required
def delete_conversations():
    try:
        session_ids = request.json.get('session_ids', [])
        if not session_ids:
            return jsonify({'status': 'error', 'message': 'No session IDs provided'}), 400
        result = chat_history_collection.delete_many({'session_id': {'$in': session_ids}})
        logger.info(f"Deleted {result.deleted_count} conversations")
        return jsonify({'status': 'success', 'message': f'Deleted {result.deleted_count} conversations'})
    except Exception as e:
        logger.error(f"Error deleting conversations: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

# API to export conversations as CSV
@app.route('/api/admin/conversations/export', methods=['GET'])
@login_required
def export_conversations():
    try:
        conversations = chat_history_collection.find()
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['Session ID', 'Role', 'Message', 'Timestamp'])
        for conv in conversations:
            session_id = conv.get('session_id', 'N/A')
            for msg in conv.get('messages', []):
                writer.writerow([
                    session_id,
                    msg.get('role', 'unknown'),
                    msg.get('message', ''),
                    msg.get('timestamp', datetime.min).isoformat()
                ])
        output.seek(0)
        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'conversations_export_{datetime.utcnow().isoformat()}.csv'
        )
    except Exception as e:
        logger.error(f"Error exporting conversations: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

# API for analytics
@app.route('/api/admin/analytics', methods=['GET'])
@login_required
def get_analytics():
    try:
        total_conversations = chat_history_collection.count_documents({})
        pipeline = [
            {'$match': {'messages': {'$exists': True, '$ne': []}}},
            {'$unwind': '$messages'},
            {'$group': {
                '_id': '$session_id',
                'message_count': {'$sum': 1}
            }},
            {'$group': {
                '_id': None,
                'avg_messages': {'$avg': '$message_count'},
                'total_messages': {'$sum': '$message_count'}
            }}
        ]
        stats = list(chat_history_collection.aggregate(pipeline))
        analytics = {
            'total_conversations': total_conversations,
            'total_messages': stats[0]['total_messages'] if stats else 0,
            'avg_messages_per_session': round(stats[0]['avg_messages'], 2) if stats else 0
        }
        return jsonify({'status': 'success', 'analytics': analytics})
    except Exception as e:
        logger.error(f"Error fetching analytics: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500
    
    
if __name__ == "__main__":
    app.run(debug=True, port=7000, host='0.0.0.0')