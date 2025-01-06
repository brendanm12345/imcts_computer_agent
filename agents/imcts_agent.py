import base64
import json
import logging
import os
import re
from typing import Dict, List, Optional, Tuple
import requests

logger = logging.getLogger("desktopenv.agent")

# Prompts
SYSTEM_PROMPT_GENERATE = """
You are an expert at simulating computer interface actions and their effects.
You will be given a screenshot of a computer interface and a task to complete. 
Your role is to suggest the N best distinct next actions that should be taken to maximize the likelihood of 
successfully completing the task.

Return ```DONE``` if task completion is detected
Return ```FAIL``` if you determine the task cannot be completed
Return ```WAIT``` if the interface needs time to respond

Format your response as:

Reflection: <analyze current state and progress toward goal>

Action Option N:
Reasoning: <detailed explanation of why this action is promising>
```python
<executable pyautogui code>
```
Imagined Future Observation T + 1:
- Immediate UI Response: <precise description of interface changes>
- Enabled Actions: <what new actions become possible>
- Progress Assessment: <how this advances toward goal>
Imagined Future Action T + 1:
- Action: <the optimal next action taken from the above imagined state>
- Reasoning: <justification for the action>

[Repeat for N distinct next best actions]

Technical requirements:
- Use only pyautogui (no screenshot() or locateCenterOnScreen())
- Provide complete code blocks
- Specify coordinates based on screenshot observation
- Click ~25px lower than visual targets
- Avoid clicking the same spot multiple times
- Default system password is 'password'
"""

SYSTEM_PROMPT_SCORE = """
You are an expert at evaluating action sequences for computer tasks.
Given a task goal and multiple recommended actions with their imagined trajectories, 
select the single best action to take next.

Respond in this exact format:
Reasoning: <1-2 sentences explaining why this is the best action>

Best action code:
```python
<paste the exact code for the best action>
```
"""


def encode_image(image_content: bytes) -> str:
    """Convert image bytes to base64 string."""
    return base64.b64encode(image_content).decode('utf-8')


def parse_code_lines(code_string: str) -> List[str]:
    """Parse Python code into executable commands."""
    executable_lines = []
    lines = code_string.split('\n')

    for line in lines:
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('import '):
            continue
        executable_lines.append(line)

    return executable_lines


class IMCTSAgent:
    def __init__(
        self,
        model: str = "gpt-4-vision-preview",
        max_tokens: int = 1500,
        temperature: float = 0.5,
        top_p: float = 0.9,
        max_trajectory_length: int = 3,
        num_actions: int = 2,
        search_depth: int = 2,
        value_threshold: float = 0.6
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.max_trajectory_length = max_trajectory_length

        # IMCTS specific parameters
        self.num_actions = num_actions
        self.search_depth = search_depth
        self.value_threshold = value_threshold

        self.thoughts: List[str] = []
        self.actions: List[str] = []
        self.observations: List[Dict] = []

    def predict(self, instruction: str, obs: Dict) -> Tuple[Optional[str], Optional[List[str]]]:
        """Main prediction loop implementing IMCTS."""
        logger.info(f"Starting new prediction for task: {instruction}")

        try:
            # Add current observation
            if "screenshot" not in obs:
                logger.error("No screenshot in observation")
                return None, None

            base64_image = encode_image(obs["screenshot"])
            self.observations.append({"screenshot": base64_image})
            # Generate actions with simulated futures
            action_response = self._generate_actions(instruction, obs)
            if not action_response:
                return None, None

            # Score and select best action
            best_code, score, reasoning = self._score_actions(
                instruction, obs, action_response)

            # Handle special responses
            if best_code in ['WAIT', 'DONE', 'FAIL']:
                return f"Special command: {best_code}", [best_code]

            # Generate thought and parse actions
            thought = f"Selected action (score: {score:.2f}). Reasoning: {
                reasoning}"
            parsed_actions = parse_code_lines(best_code)

            if not parsed_actions:
                return None, None

            # Update state
            self.thoughts.append(thought)
            self.actions.append(best_code)

            return thought, parsed_actions

        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            return None, None

    def _generate_actions(self, instruction: str, obs: Dict) -> Optional[str]:
        """Generate diverse candidate actions with imagined trajectories."""
        messages = []

        # System message
        messages.append({
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": f"{SYSTEM_PROMPT_GENERATE}\nYou are asked to complete the following task: {instruction}"
                }
            ]
        })

        # Add history if available
        if len(self.observations) > self.max_trajectory_length:
            history_observations = self.observations[-self.max_trajectory_length:]
            history_actions = self.actions[-self.max_trajectory_length:]
            history_thoughts = self.thoughts[-self.max_trajectory_length:]
        else:
            history_observations = self.observations
            history_actions = self.actions
            history_thoughts = self.thoughts

        # Add trajectory context
        for prev_obs, prev_action, prev_thought in zip(
            history_observations, history_actions, history_thoughts
        ):
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
                            "url": f"data:image/png;base64,{prev_obs['screenshot']}",
                            "detail": "high"
                        }
                    }
                ]
            })

            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": prev_thought}]
            })

        # Add current request
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Please suggest N = {self.num_actions} distinct next best actions and simulate K = {self.search_depth} future steps for each."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encode_image(obs['screenshot'])}",
                        "detail": "high"
                    }
                }
            ]
        })

        try:
            # Higher values for diversity
            return self._call_llm(messages, temperature=1.0, top_p=0.95)
        except Exception as e:
            logger.error(f"Error generating actions: {str(e)}")
            return None

    def _score_actions(self, instruction: str, obs: Dict, actions_response: str) -> Tuple[str, float, str]:
        """Score action trajectories and select the best one."""
        if actions_response.strip() in ['WAIT', 'DONE', 'FAIL']:
            return actions_response.strip(), 1.0, "Special command"

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT_SCORE}]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Task Goal: {instruction}\n\nRecommended Actions with their imagined trajectories:\n{actions_response}"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{encode_image(obs['screenshot'])}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ]

        try:
            response = self._call_llm(messages, temperature=0.3)

            # Extract code and reasoning
            code_blocks = re.findall(
                r'```python\s*(.*?)\s*```', response, re.DOTALL)
            best_code = code_blocks[-1].strip() if code_blocks else ""

            reasoning_match = re.search(
                r'Reasoning:\s*(.*?)(?=\n\nBest action code:|$)', response, re.DOTALL)
            reasoning = reasoning_match.group(1).strip(
            ) if reasoning_match else "No reasoning provided"

            if not best_code:
                return "", 0.0, "No valid code found"

            return best_code, 1.0, reasoning

        except Exception as e:
            logger.error(f"Error in action selection: {str(e)}")
            return "", 0.0, f"Error in selection: {str(e)}"

    def _call_llm(self, messages: List[Dict], temperature: Optional[float] = None, top_p: Optional[float] = None) -> Optional[str]:
        """Call the appropriate LLM API based on model name."""
        if self.model.startswith("gpt"):
            return self._call_openai_api(messages, temperature, top_p)
        elif self.model.startswith("claude"):
            return self._call_anthropic_api(messages, temperature, top_p)
        else:
            raise ValueError(f"Unsupported model: {self.model}")

    def _call_openai_api(self, messages: List[Dict], temperature: Optional[float] = None, top_p: Optional[float] = None) -> Optional[str]:
        """Call OpenAI API with proper formatting."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": temperature or self.temperature,
            "top_p": top_p or self.top_p
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

    def _call_anthropic_api(self, messages: List[Dict], temperature: Optional[float] = None, top_p: Optional[float] = None) -> Optional[str]:
        """Call Anthropic API with proper formatting."""
        anthropic_messages = []

        for message in messages:
            anthropic_message = {
                "role": message["role"],
                "content": []
            }

            # Handle both string content and array content
            message_content = message.get("content", [])
            if isinstance(message_content, str):
                anthropic_message["content"].append({
                    "type": "text",
                    "text": message_content
                })
            else:
                for content in message_content:
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
            "temperature": temperature or self.temperature,
            "top_p": top_p or self.top_p
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
