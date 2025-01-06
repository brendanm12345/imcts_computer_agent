import base64
import json
import logging
import re
from typing import Dict, List, Optional, Tuple
import requests
import os

logger = logging.getLogger("desktopenv.agent")

# OS World Original Prompt
SYS_PROMPT_IN_SCREENSHOT_OUT_CODE = """
You are an agent which follow my instruction and perform desktop computer tasks as instructed.
You have good knowledge of computer and good internet connection and assume your code will run on a computer for controlling the mouse and keyboard.
For each step, you will get an observation of an image, which is the screenshot of the computer screen and you will predict the action of the computer based on the image.

You are required to use `pyautogui` to perform the action grounded to the observation, but DONOT use the `pyautogui.locateCenterOnScreen` function to locate the element you want to operate with since we have no image of the element you want to operate with. DONOT USE `pyautogui.screenshot()` to make screenshot.
Return one line or multiple lines of python code to perform the action each time, be time efficient. When predicting multiple lines of code, make some small sleep like `time.sleep(0.5);` interval so that the machine could take; Each time you need to predict a complete code, no variables or function can be shared from history
You need to to specify the coordinates of by yourself based on your observation of current observation, but you should be careful to ensure that the coordinates are correct.
You ONLY need to return the code inside a code block, like this:
```python
# your code here
```
Specially, it is also allowed to return the following special code:
When you think you have to wait for some time, return ```WAIT```;
When you think the task can not be done, return ```FAIL```, don't easily say ```FAIL```, try your best to do the task;
When you think the task is done, return ```DONE```.

My computer's password is 'password', feel free to use it when you need sudo rights.
First give the current screenshot and previous things we did a short reflection, then RETURN ME THE CODE OR SPECIAL CODE I ASKED FOR. NEVER EVER RETURN ME ANYTHING ELSE.
""".strip()


def encode_image(image_content: bytes) -> str:
    """Convert image bytes to base64 string."""
    return base64.b64encode(image_content).decode('utf-8')


def parse_code_from_string(input_string: str) -> List[str]:
    """Extract code blocks and special commands from the response."""
    input_string = "\n".join(
        [line.strip() for line in input_string.split(';') if line.strip()]
    )

    # Handle special commands
    if input_string.strip() in ['WAIT', 'DONE', 'FAIL']:
        return [input_string.strip()]

    # Extract code blocks
    pattern = r"```(?:python\s+)?(.*?)```"
    matches = re.findall(pattern, input_string, re.DOTALL)

    codes = []
    for match in matches:
        match = match.strip()
        if match in ['WAIT', 'DONE', 'FAIL']:
            codes.append(match)
        elif match.split('\n')[-1] in ['WAIT', 'DONE', 'FAIL']:
            if len(match.split('\n')) > 1:
                codes.append("\n".join(match.split('\n')[:-1]))
            codes.append(match.split('\n')[-1])
        else:
            codes.append(match)

    return codes


class PromptAgent:
    def __init__(
        self,
        model: str = "gpt-4-vision-preview",
        max_tokens: int = 1500,
        temperature: float = 0.5,
        top_p: float = 0.9,
        max_trajectory_length: int = 3,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.max_trajectory_length = max_trajectory_length

        self.thoughts: List[str] = []
        self.actions: List[str] = []
        self.observations: List[Dict] = []

    def predict(self, instruction: str, obs: Dict) -> Tuple[Optional[str], Optional[List[str]]]:
        """Generate next action based on instruction and observation."""
        system_message = f"{
            SYS_PROMPT_IN_SCREENSHOT_OUT_CODE}\nYou are asked to complete the following task: {instruction}"

        # Format messages for API calls
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}]
            }
        ]

        # Add history if available
        if len(self.observations) > self.max_trajectory_length:
            history_observations = self.observations[-self.max_trajectory_length:]
            history_actions = self.actions[-self.max_trajectory_length:]
            history_thoughts = self.thoughts[-self.max_trajectory_length:]
        else:
            history_observations = self.observations
            history_actions = self.actions
            history_thoughts = self.thoughts

        for prev_obs, prev_action, prev_thought in zip(
            history_observations, history_actions, history_thoughts
        ):
            # Add previous observation
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Given the screenshot below, what's the next step?"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{prev_obs['screenshot']}"
                        }
                    }
                ]
            })

            # Add previous response
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": prev_thought}]
            })

        # Add current observation
        base64_image = encode_image(obs["screenshot"])
        self.observations.append({"screenshot": base64_image})

        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Given the screenshot below, what's the next step?"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                }
            ]
        })

        try:
            response = self._call_llm(messages)
            if response is None:
                return None, None

            response_text = response
            actions = parse_code_from_string(response_text)

            self.thoughts.append(response_text)
            self.actions.append(actions)

            return response_text, actions

        except Exception as e:
            logger.error(f"API error: {type(e).__name__} - {str(e)}")
            return None, None

    # TODO: refactor to use liteLLM
    def _call_llm(self, messages: List[Dict]) -> Optional[str]:
        """Call the appropriate LLM API based on model name. Supported providers include OpenAI and Anthropic"""
        if self.model.startswith("gpt"):
            return self._call_openai_api(messages)
        elif self.model.startswith("claude"):
            return self._call_anthropic_api(messages)
        else:
            raise ValueError(f"Unsupported model: {self.model}")

    def _call_openai_api(self, messages: List[Dict]) -> Optional[str]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p
        }

        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )

        if response.status_code != 200:
            logger.error(f"OpenAI API error: {response.text}")
            return None

        return response.json()['choices'][0]['message']['content']

    def _call_anthropic_api(self, messages: List[Dict]) -> Optional[str]:
        anthropic_messages = []

        for message in messages:
            anthropic_message = {
                "role": message["role"],
                "content": []
            }

            for content in message["content"]:
                if content["type"] == "text":
                    anthropic_message["content"].append({
                        "type": "text",
                        "text": content["text"]
                    })
                elif content["type"] == "image_url":
                    image_data = content["image_url"]["url"].replace(
                        "data:image/png;base64,", ""
                    )
                    anthropic_message["content"].append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_data
                        }
                    })

            anthropic_messages.append(anthropic_message)

        # Handle system message for Anthropic
        if anthropic_messages[0]["role"] == "system":
            system_content = anthropic_messages[0]["content"][0]
            anthropic_messages[1]["content"].insert(0, system_content)
            anthropic_messages.pop(0)

        headers = {
            "x-api-key": os.environ["ANTHROPIC_API_KEY"],
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p
        }

        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload
        )

        if response.status_code != 200:
            logger.error(f"Anthropic API error: {response.text}")
            return None

        return response.json()['content'][0]['text']

    def reset(self):
        """Reset agent state."""
        self.thoughts = []
        self.actions = []
        self.observations = []
