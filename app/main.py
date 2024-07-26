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


client_manager = ClientManager()
client_manager.init_client(
    inference_port=INFERENCE_PORT,
    host=INFERENCE_HOST,
    custom_tools=[],
    disable_safety=DISABLE_SAFETY,
)


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

@me.api("/api/chat", methods=["POST"])
def api_chat():
    data = request.json
    user_id = data.get('user_id')
    message = data.get('message')
    
    if not user_id or not message:
        return jsonify({"error": "Missing user_id or message"}), 400
    
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
    import subprocess

    subprocess.run(["mesop", __file__])
