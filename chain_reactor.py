import os
import argparse
import time
import uuid
import json
import copy
import datetime
import logging
import re
from typing import List
from flask import Flask, request, jsonify, Response
from llama_index.llms.ollama import Ollama
from llama_index.core.llama_pack.base import BaseLlamaPack
from llama_index.core.llms.llm import LLM
from llama_index.core.llms import ChatMessage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class ChainReactor(BaseLlamaPack):
    def __init__(
            self,
            reactor_llms: List[LLM],
            controller_llm: LLM,
            iterations: int = 1,
            max_tokens: int = 128000,
            temperature: float = 0.7,
            output_dir: str = "./chain_reactor_output",
            aider_mode: bool = False,
    ) -> None:
        self.reactor_llms = reactor_llms
        self.controller_llm = controller_llm
        self.iterations = iterations
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.output_dir = output_dir
        self.aider_mode = aider_mode
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def strip_assistant_prefix(response: str) -> str:
        if response.startswith("assistant:"):
            response = response[len("assistant:"):].strip()
        return response

    def save_to_file(self, content: str, filename: str) -> None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        prefixed_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(self.output_dir, prefixed_filename)
        with open(filepath, "w") as file:
            file.write(content)

    def detect_fence(self, system_prompt: str) -> tuple:
        fences = {
            "source": self.wrap_fence("source"),
            "code": self.wrap_fence("code"),
            "pre": self.wrap_fence("pre"),
            "codeblock": self.wrap_fence("codeblock"),
            "sourcecode": self.wrap_fence("sourcecode"),
        }

        for fence_name in fences.keys():
            if f"<{fence_name}>" in system_prompt.lower():
                return fences[fence_name]

        # Default to triple backticks if no match is found
        return "``" + "`", "``" + "`"

    @staticmethod
    def wrap_fence(name):
        return f"<{name}>", f"</{name}>"

    def request_and_response(
            self,
            reactor_llm: LLM,
            messages: List[ChatMessage],
            iteration: int,
            agent_index: int,
            system_prompt: str = "",
            previous_response: str = "",
            is_first_pass: bool = True
    ) -> str:
        messages = copy.deepcopy(messages)

        if is_first_pass:
            messages.insert(0, ChatMessage(role="system", content=system_prompt))
        else:
            enhanced_system_message = (
                "Refine the previous response based on the user prompt provided. "
                "The previous response is enclosed between '---START PREVIOUS RESPONSE---' and '---END PREVIOUS RESPONSE---'. "
                "Focus solely on the request. For coding tasks, fix any bugs or errors in the code. "
                "Provide the refined code or solution without adding any explanations or confirmations. "
                "If no changes are needed, repeat the previous response exactly as it is, with no additional text. "
                "Avoid any commentary such as 'The code is correct' or 'No changes are needed'. "
                "Ensure the output does not include '---START PREVIOUS RESPONSE---' and '---END PREVIOUS RESPONSE---'."
            )

            if self.aider_mode:
                enhanced_system_message += (
                    " Preserve the search line '<<<<<<< SEARCH', dividing line '=======', "
                    "and replace line '>>>>>>> REPLACE' if they are present."
                )

            combined_system_prompt = f"{enhanced_system_message}\n\n{system_prompt}"

            messages.insert(0, ChatMessage(role="system", content=combined_system_prompt))
            messages.append(ChatMessage(role="user", content=f"---START PREVIOUS RESPONSE---\n{previous_response}\n---END PREVIOUS RESPONSE---"))

        input_content = "\n".join([f"{msg.role}: {msg.content}" for msg in messages])
        self.save_to_file(input_content, f"iteration_{iteration}_agent_{agent_index}_input.txt")

        response = str(reactor_llm.chat(messages, max_tokens=self.max_tokens, temperature=self.temperature)).strip()
        response = self.strip_assistant_prefix(response)

        self.save_to_file(response, f"iteration_{iteration}_agent_{agent_index}_output.txt")

        return response

    def compare_responses(
            self,
            system_prompt: str,
            all_messages: List[ChatMessage],
            original_response: str,
            new_response: str
    ) -> str:
        if self.aider_mode:
            # Detect the fence from the system prompt
            open_fence, close_fence = self.detect_fence(system_prompt)

            # Regular expression to detect individual fenced blocks
            block_pattern = re.compile(
                rf"{re.escape(open_fence)}(.*?){re.escape(close_fence)}",
                re.DOTALL
            )

            # Regular expression to validate the full search and replace pattern within a block
            full_pattern = re.compile(
                r"<<<<<<< SEARCH.*?=======.*?>>>>>>> REPLACE",
                re.DOTALL
            )

            def is_fully_correct(response: str) -> bool:
                blocks = block_pattern.findall(response)
                # A response is only fully correct if all detected blocks are valid
                return all(full_pattern.search(block) for block in blocks)

            original_fully_correct = is_fully_correct(original_response)
            new_fully_correct = is_fully_correct(new_response)

            # Reject responses that have some but not all patterns correct
            if new_fully_correct and not original_fully_correct:
                return new_response
            elif original_fully_correct and not new_fully_correct:
                return original_response
            else:
                # Fall back to comparison LLM if neither response is fully correct
                pass  # Continue to use the comparison logic below

        comparison_prompt = (
            "You will be given a user prompt, an original response, and a new response. "
            "Evaluate the original and new responses based on how well they match the user prompt. "
            "For coding tasks, choose the response that more accurately and completely implements the requirements specified in the user prompt. "
            "Respond with only 'original' or 'new'. "
            "Do not include any additional text, explanations, or commentary."
        )

        user_prompt_content = all_messages[-1].content

        # Creating the full comparison prompt
        user_prompt = (
            "---START USER PROMPT---\n"
            f"{user_prompt_content}\n"
            "---END USER PROMPT---\n\n"
            "---START ORIGINAL RESPONSE---\n"
            f"{original_response}\n"
            "---END ORIGINAL RESPONSE---\n\n"
            "---START NEW RESPONSE---\n"
            f"{new_response}\n"
            "---END NEW RESPONSE---"
        )

        messages = [
            ChatMessage(role="system", content=comparison_prompt),
            ChatMessage(role="user", content=user_prompt),
        ]

        input_content = "\n".join([f"{msg.role}: {msg.content}" for msg in messages])
        self.save_to_file(input_content, "controller_input.txt")

        comparison_result = str(self.controller_llm.chat(messages, max_tokens=self.max_tokens, temperature=self.temperature)).strip()
        comparison_result = self.strip_assistant_prefix(comparison_result)

        self.save_to_file(comparison_result, "controller_output.txt")

        return original_response if "original" in comparison_result.lower() else new_response


    def run(self, system_prompt: str, all_messages: List[ChatMessage]) -> str:
        previous_response = ""

        for iteration in range(self.iterations):
            is_first_pass = iteration == 0

            for agent_index, reactor_llm in enumerate(self.reactor_llms):
                messages = all_messages.copy()

                new_response = ""
                try:
                    new_response = self.request_and_response(
                        reactor_llm=reactor_llm,
                        messages=messages,
                        iteration=iteration,
                        agent_index=agent_index,
                        system_prompt=system_prompt,
                        previous_response=previous_response,
                        is_first_pass=is_first_pass
                    )
                except Exception as e:
                    logger.error("Error request and response: %s", e)
                    continue

                if previous_response:
                    try:
                        previous_response = self.compare_responses(
                            system_prompt=system_prompt,
                            all_messages=messages,
                            original_response=previous_response,
                            new_response=new_response
                        )
                    except Exception as e:
                        logger.error("Error comparing responses: %s", e)
                        continue
                else:
                    previous_response = new_response

                is_first_pass = False

        return previous_response

@app.route('/v1/models', methods=['GET'])
def list_models():
    return jsonify({
        "object": "list",
        "data": [
            {"id": "chain-reactor", "object": "model"}
        ]
    })

@app.route('/v1/chat/completions', methods=['POST'])
def create_completion():
    try:
        data = request.json
        logger.info("Request JSON: %s", data)

        system_prompt = ""
        all_messages = []

        for message in data['messages']:
            if message['role'] == 'system':
                system_prompt = message['content']
            else:
                all_messages.append(ChatMessage(role=message['role'], content=message['content']))

        response = chain_reactor.run(system_prompt=system_prompt, all_messages=all_messages)
        completion_id = str(uuid.uuid4())

        if data.get('stream', False):
            def generate():
                for chunk in response.split('\n'):
                    chunk_data = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": "chain-reactor",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": f"{chunk}\n"},
                            }
                        ]
                    }
                    logger.debug("Streaming chunk: %s", chunk_data)
                    yield f"data: {json.dumps(chunk_data)}\n\n"
                logger.info("Streaming completed")
                yield f"data: [DONE]\n\n"

            return Response(generate(), content_type='text/event-stream')

        else:
            response_payload = {
                "id": completion_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "chain-reactor",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response
                        },
                        "finish_reason": "stop",
                        "logprobs": None
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            }

            logger.info("Response payload: %s", response_payload)
            return jsonify(response_payload)

    except Exception as e:
        logger.error("Error processing request: %s", e)
        return jsonify({"error": str(e)}), 500

def main():
    parser = argparse.ArgumentParser(description="ChainReactor CLI for iterative LLM processing.")
    parser.add_argument("--system-prompt", type=str, help="System prompt to guide the LLMs.")
    parser.add_argument("--user-prompt", type=str, help="User prompt to be processed by the LLMs.")
    parser.add_argument("--serve", action="store_true", help="Run in server mode.")
    parser.add_argument("--aider", action="store_true", help="Enable aider mode for additional formatting of output.")
    args = parser.parse_args()

    global chain_reactor
    chain_reactor = ChainReactor(
        reactor_llms=[
            # Ollama(model="llama3.1:8b-instruct-fp16", request_timeout=300.0),
            # Ollama(model="mistral-nemo:12b-instruct-2407-q8_0", request_timeout=300.0),
            Ollama(model="codestral:22b-v0.1-q8_0", request_timeout=300.0),
        ],
        controller_llm=Ollama(model="phi3:14b-medium-128k-instruct-q8_0", request_timeout=300.0),
        iterations=2,
        max_tokens=128000,
        temperature=0.7,
        aider_mode=args.aider
    )

    if args.serve:
        app.run(host='0.0.0.0', port=5000)
    else:
        user_message = ChatMessage(role="user", content=args.user_prompt)
        all_messages = [user_message]
        response = chain_reactor.run(system_prompt=args.system_prompt, all_messages=all_messages)
        print(response)

if __name__ == '__main__':
    main()
