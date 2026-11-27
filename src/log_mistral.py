"""Optional demo: Mistral function-calling (get_order_status tool) traced with
mlflow.mistral.autolog() into experiment 'Mistral'.

Requires the MISTRAL_API_KEY environment variable.

Usage:
    uv run python src/log_mistral.py
"""
import json
import os

import mlflow
from mistralai.client import Mistral

from heal_mlflow_showcase import quiet_logger
from heal_mlflow_showcase.config import TRACKING_URI

# set an environmental variable for your mistral key
client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

quiet_logger()
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("Mistral")
mlflow.mistral.autolog()

# CODE SOURCE -> https://docs.mistral.ai/inference/quickstarts/call-your-own-functions

def get_order_status(order_id: str) -> dict:
    """Your own code. Here, a lookup in a small table."""
    orders = {"A-1042": "shipped", "A-1043": "in preparation"}
    return {"order_id": order_id, "status": orders.get(order_id, "unknown")}

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "Get the delivery status of an order.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "The order ID, for example A-1042."},
                },
                "required": ["order_id"],
            },
        },
    }
]
with mlflow.start_run():
    
    messages = [{"role": "user", "content": "Where is my order A-1042?"}]

    # 1. The model reads the question and asks for get_order_status.
    response = client.chat.complete(model="mistral-small-latest", messages=messages, tools=tools)
    assistant = response.choices[0].message
    messages.append(assistant.model_dump(exclude_unset=True))

    # 2. Your code runs the function and adds its result to the conversation.
    for call in assistant.tool_calls or []:
        raw = call.function.arguments
        args = json.loads(raw) if isinstance(raw, str) else raw
        print(f"Model asked for: {call.function.name}({args})")
        result = get_order_status(**args)
        messages.append(
            {
                "role": "tool",
                "name": call.function.name,
                "tool_call_id": call.id,
                "content": json.dumps(result),
            }
        )

    # 3. The model answers the question with the result.
    final = client.chat.complete(model="mistral-small-latest", messages=messages, tools=tools)
    print(final.choices[0].message.content)
