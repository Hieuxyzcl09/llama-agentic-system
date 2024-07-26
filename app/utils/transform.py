# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import uuid

import mesop as me
from llama_agentic_system.api.datatypes import (
    AgenticSystemTurnResponseEventType,
    StepType,
)

from .chat import KEY_TO_OUTPUTS, MAX_VIOLATIONS, State, StepStatus

from .client import ClientManager
from .common import sync_generator

from llama_toolchain.inference.api import *  # noqa: F403


EVENT_LOOP = asyncio.new_event_loop()


EventType = AgenticSystemTurnResponseEventType


def transform(content: InterleavedTextAttachment, chat_history=None):
    state = me.state(State)

    # Nếu có lịch sử chat, chuyển đổi nó thành định dạng phù hợp
    messages = []
    if chat_history:
        for entry in chat_history:
            if entry['role'] == 'user':
                messages.append(UserMessage(content=entry['content']))
            elif entry['role'] == 'assistant':
                messages.append(CompletionMessage(content=entry['content']))

    # Thêm tin nhắn hiện tại vào danh sách
    messages.append(UserMessage(content=content))

    client_manager = ClientManager()
    client = client_manager.get_client()

    generator = sync_generator(EVENT_LOOP, client.run(messages))
    response_content = ""
    for chunk in generator:
        if not hasattr(chunk, "event"):
            if isinstance(chunk, ToolResponseMessage):
                yield str(uuid.uuid4()), chunk
            continue

        event = chunk.event
        event_type = event.payload.event_type
        if event_type == EventType.turn_start.value:
            continue

        if event_type == EventType.step_start.value:
            continue

        elif event_type == EventType.step_progress.value:
            step_type = event.payload.step_type
            step_uuid = event.payload.step_id

            if step_type == StepType.inference:
                existing = KEY_TO_OUTPUTS.get(step_uuid)
                if event.payload.tool_call_delta:
                    delta = event.payload.tool_call_delta
                    if existing:
                        if isinstance(delta.content, str):
                            existing.content += delta.content
                            response_content += delta.content

                        if delta.parse_status == ToolCallParseStatus.failure:
                            existing.content = (
                                "<b><<Tool call parsing FAILED!>></b> "
                                + existing.content
                            )

                        yield step_uuid, existing
                    else:
                        yield step_uuid, StepStatus(
                            step_type=step_type, content=delta.content
                        )
                else:
                    if existing and isinstance(existing, CompletionMessage):
                        existing.content += event.payload.model_response_text_delta
                        response_content += event.payload.model_response_text_delta
                        yield step_uuid, existing
                    else:
                        yield step_uuid, CompletionMessage(
                            content=event.payload.model_response_text_delta,
                            stop_reason=StopReason.end_of_turn,
                        )
                        response_content += event.payload.model_response_text_delta
            elif step_type == StepType.tool_execution:
                pass

        elif event_type == EventType.step_complete.value:
            step_type = event.payload.step_type
            step_uuid = event.payload.step_details.step_id

            if step_type == StepType.shield_call:
                response = event.payload.step_details.response
                if response.is_violation:
                    state.violation_count += 1
                    if state.moderated:
                        if state.violation_count >= MAX_VIOLATIONS:
                            chat_moderation_message = f"You violated safety guidelines {MAX_VIOLATIONS} times. Chat closing."
                        else:
                            chat_moderation_message = f"Your message violated safety guidelines. So far you've made {state.violation_count} violations. After {MAX_VIOLATIONS} violations this chat will close."

                        yield step_uuid, CompletionMessage(
                            content=chat_moderation_message,
                            stop_reason=StopReason.end_of_turn,
                        )
                        response_content += chat_moderation_message
                    elif response.violation_return_message:
                        yield step_uuid, CompletionMessage(
                            content=response.violation_return_message,
                            stop_reason=StopReason.end_of_turn,
                        )
                        response_content += response.violation_return_message
                    else:
                        yield step_uuid, StepStatus(
                            step_type=StepType.shield_call,
                            content=response,
                        )
                else:
                    yield step_uuid, StepStatus(
                        step_type=StepType.shield_call,
                        content=response,
                    )
                continue
            elif step_type == StepType.tool_execution:
                details = event.payload.step_details
                response = details.tool_responses[0]

                yield step_uuid, StepStatus(
                    step_type=StepType.tool_execution,
                    content=response.content,
                )
                response_content += response.content

    return response_content
