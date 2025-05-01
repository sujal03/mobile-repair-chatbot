from flask import Flask, jsonify, render_template, request, redirect, url_for, session, abort
from pymongo import MongoClient
from functools import wraps
from datetime import datetime
import os
import logging
import hashlib
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your-secret-key')  # Use env var in production

# MongoDB setup
mongo_client = MongoClient(os.getenv('MONGO_URI'))
db = mongo_client['mobile_repair_bot']
chat_history_collection = db['chat_history']

# Admin credentials (hardcoded for simplicity; use env vars or DB in production)
ADMIN_USERNAME = 'admin'
ADMIN_PASSWORD_HASH = hashlib.sha256('admin123'.encode()).hexdigest()

# Login required decorator
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
        password = request.form.get('password')
        if (username == ADMIN_USERNAME and 
            hashlib.sha256(password.encode()).hexdigest() == ADMIN_PASSWORD_HASH):
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

# Route to serve admin panel
@app.route('/admin')
@login_required
def admin_panel():
    return render_template('admin.html')

# API to get list of conversations
@app.route('/api/admin/conversations', methods=['GET'])
@login_required
def get_conversations():
    try:
        # Aggregate conversations by session_id, get latest message
        pipeline = [
            {
                '$match': {'session_id': {'$exists': True}, 'messages': {'$exists': True, '$ne': []}}
            },
            {
                '$unwind': '$messages'  # Flatten the messages array
            },
            {
                '$sort': {'messages.timestamp': -1}  # Sort by message timestamp descending
            },
            {
                '$group': {
                    '_id': '$session_id',
                    'latest_message': {'$first': {'$ifNull': ['$messages.message', '']}},
                    'latest_role': {'$first': {'$ifNull': ['$messages.role', 'unknown']}},
                    'latest_timestamp': {'$first': {'$ifNull': ['$messages.timestamp', datetime.min]}},
                    'message_count': {'$sum': 1}
                }
            },
            {
                '$sort': {'latest_timestamp': -1}  # Sort conversations by latest message
            },
            {
                '$limit': 100  # Limit to 100 conversations for performance
            }
        ]
        conversations = list(chat_history_collection.aggregate(pipeline))
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
            ]
        })
    except Exception as e:
        logger.error(f"Error fetching conversations: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

# API to get full conversation by session_id
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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)