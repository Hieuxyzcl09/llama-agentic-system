# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import mesop as me

from utils.chat import chat, State
from utils.client import ClientManager
from utils.common import DISABLE_SAFETY, INFERENCE_HOST, INFERENCE_PORT, on_attach
from utils.transform import transform
from flask import request, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
import secrets
import os

client_manager = ClientManager()
client_manager.init_client(
    inference_port=INFERENCE_PORT,
    host=INFERENCE_HOST,
    custom_tools=[],
    disable_safety=DISABLE_SAFETY,
)

basedir = os.path.abspath(os.path.dirname(__file__))
me.app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'apikeys.db')
me.app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(me.app)

class APIKey(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.String(64), unique=True, nullable=False)
    user_id = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    expires_at = db.Column(db.DateTime)
    is_active = db.Column(db.Boolean, default=True)

@me.page(
    path="/",
    title="Llama Agentic System",
)
def page():
    state = me.state(State)
    chat(
        transform,
        title="Llama Agentic System",
        bot_user="Llama Agent",
        on_attach=on_attach,
    )

@me.api("/api/generate_api_key", methods=["POST"])
def generate_api_key():
    data = request.json
    user_id = data.get('user_id')
    if not user_id:
        return jsonify({"error": "User ID is required"}), 400

    new_key = secrets.token_urlsafe(32)
    expires_at = datetime.utcnow() + timedelta(days=30)

    api_key = APIKey(key=new_key, user_id=user_id, expires_at=expires_at)
    db.session.add(api_key)
    db.session.commit()

    return jsonify({"api_key": new_key, "expires_at": expires_at.isoformat()}), 201

@me.api("/api/revoke_api_key", methods=["POST"])
def revoke_api_key():
    data = request.json
    key = data.get('api_key')
    if not key:
        return jsonify({"error": "API key is required"}), 400

    api_key = APIKey.query.filter_by(key=key).first()
    if not api_key:
        return jsonify({"error": "Invalid API key"}), 404

    api_key.is_active = False
    db.session.commit()

    return jsonify({"message": "API key revoked successfully"}), 200

@me.api("/api/list_api_keys", methods=["GET"])
def list_api_keys():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({"error": "User ID is required"}), 400

    api_keys = APIKey.query.filter_by(user_id=user_id).all()
    keys_list = [
        {
            "key": key.key,
            "created_at": key.created_at.isoformat(),
            "expires_at": key.expires_at.isoformat(),
            "is_active": key.is_active
        }
        for key in api_keys
    ]

    return jsonify(keys_list)

@me.api("/api/chat", methods=["POST"])
def api_chat():
    data = request.json
    user_id = data.get('user_id')
    message = data.get('message')
    api_key = data.get('api_key')
    
    if not user_id or not message or not api_key:
        return jsonify({"error": "Missing user_id, message, or api_key"}), 400
    
    # Kiểm tra API key
    key = APIKey.query.filter_by(key=api_key, user_id=user_id, is_active=True).first()
    if not key or key.expires_at < datetime.utcnow():
        return jsonify({"error": "Invalid or expired API key"}), 401
    # Lấy lịch sử chat của người dùng
    user_history = chat_history.get(user_id, [])
    
    # Thêm tin nhắn mới vào lịch sử
    user_history.append({"role": "user", "content": message})
    
    # Gọi hàm transform để xử lý tin nhắn và lịch sử chat
    response = transform(message, user_history)
    
    # Thêm phản hồi vào lịch sử
    user_history.append({"role": "assistant", "content": response})
    
    # Lưu lịch sử cập nhật
    chat_history[user_id] = user_history
    
    return jsonify({"response": response})

if __name__ == "__main__":
    with me.app.app_context():
        db.create_all()
    import subprocess
    subprocess.run(["mesop", __file__])
