"""
Chat handler

"""

import json
import logging
import os
import pathlib
import random
import textwrap
from typing import Literal

import pydantic
from openai import OpenAI
from openai.types.chat import ChatCompletion
from openai.types.chat import ChatCompletionMessageParam as OpenAIMessageParam
from anthropic import Anthropic
from anthropic.types import MessageParam as AnthropicMessageParam
from anthropic.types import Message as AnthropicMessage


Model = Literal[
    "gpt-4-turbo-2024-04-09",
    "claude-3-opus-20240229",
]


class Agent(pydantic.BaseModel):
    """LLM agent"""

    client: OpenAI | Anthropic
    model: str

    def __init__(self, model: Model, **data):
        super().__init__(model=model, **data)
        self.model = model
        if model.startswith("gpt-4"):
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key is None:
                raise ValueError("Failed to get env $OPENAI_API_KEY")
            self.client = OpenAI(api_key=api_key)
        elif model.startswith("claude"):
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if api_key is None:
                raise ValueError("Failed to get env $ANTHROPIC_API_KEY")
            self.client = Anthropic(api_key=api_key)
        else:
            # raise error
            raise ValueError("Invalid model specified.")

    def generate(self, prompt: str, prefix: str | None = None) -> str:
        """Generate a response based on the prompt."""
        if isinstance(self.client, OpenAI):
            msgs: list[OpenAIMessageParam] = [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
            resp: ChatCompletion = self.client.chat.completions.create(
                messages=msgs,
                model=self.model,
                max_tokens=1024,
                temperature=0,
            )
            res = resp.choices[0].message.content
            if res is None:
                raise ValueError("Failed to get a response from OpenAI LLM")
            return res

        elif isinstance(self.client, Anthropic):
            messages: list[AnthropicMessageParam] = [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
            if prefix:
                response_message: AnthropicMessageParam = {
                    "role": "assistant",
                    "content": prefix,
                }
                messages.append(response_message)
            response: AnthropicMessage = self.client.messages.create(
                messages=messages,
                model=self.model,
                max_tokens=1024,
                temperature=0.0,
            )
            return response.content[0].text
        else:
            raise ValueError("Invalid client type.")


class User(pydantic.BaseModel):
    """User as a chart participant."""

    character: str
    role: str
    agent: Agent

    def generate(self, prompt: str, prefix: str | None = None) -> str:
        """Get a LLM response"""
        return self.agent.generate(prompt, prefix)


class GameMaster(pydantic.BaseModel):
    """Game master controlling who to post next."""

    strategy: str
    agent: Agent

    def generate(self, prompt: str, prefix: str | None = None) -> str:
        """Get a LLM response"""
        return self.agent.generate(prompt, prefix)


class System(pydantic.BaseModel):
    """Chat system"""

    gamemaster: GameMaster
    users: list[User]


class Post(pydantic.BaseModel):
    """Post in a thread."""

    id: int
    username: str
    text: str
    in_reply_to: list[int] = []


class Thread(pydantic.BaseModel):
    """A discussion thread"""

    id: int
    theme: str
    posts: list[Post]


def load_users() -> list[User]:
    """Load users from the JSON data file."""
    p = pathlib.Path("data/users.json")
    with p.open("r", encoding="utf8") as f:
        users = []
        for userdata in json.load(f):
            agent = Agent(userdata["model"])
            user = User(**userdata, agent=agent)
            users.append(user)
    return users


def load_gamemaster() -> GameMaster:
    """Load the game master from the JSON data file."""
    p = pathlib.Path("data/strategies.json")
    with p.open("r", encoding="utf8") as f:
        agent = Agent("gpt-4-turbo-2024-04-09")  # hardcoded for now
        gamemaster = GameMaster(**json.load(f), agent=agent)
    return gamemaster


def select_user(system: System, thread: Thread) -> User:
    """Select a user posting next"""
    prompt = _get_user_selection_prompt(system=system, thread=thread)
    response = system.gamemaster.generate(prompt, prefix="User: ")
    if not response or not response.startswith("User: "):
        raise ValueError(f"LLM behaving weird: {response}")

    r = response.lower()
    for user in system.users:
        if user.character.lower() in r:
            selected = user
            break
    else:
        selected = random.choice(system.users)
        logging.info(f"Choosing random user: {user.character}")

    return selected


def gen_post(user: User, thread: Thread) -> Post:
    """Generate a post for the user."""
    id_ = thread.posts[-1].id + 1 if thread.posts else 1
    prompt = _get_user_prompt(user, thread)
    text = user.generate(prompt)
    if text is None:
        raise ValueError("Failed to generate text.")
    return Post(id=id_, username=user.character, text=text)


def _get_user_prompt(user: User, thread: Thread) -> str:
    """Generate a prompt for user content generation"""
    s = f"""
        ### Objective

        Post a comment to this given discussion thread:

        <thread>
        {thread}
        </thread>


        ### Username and Role

        username: {user.character}
        role: {user.role}

        ### Thread format
        The discussion is in the form of temporally-ordered messages. You may use "->" in the first line to show reply explicitly to one or more posts. Participants are anonymous and no usernames is displayed. Keep a comment 1 to 5 senstences long.

        <example>
        [1]
        Hey fellow patient gamers! I'm looking for some cheap and fun games on Steam that are worth the wait. Any recommendations?

        [2]
        Stardew Valley is a must-play! It's a charming farming RPG with tons of content and replayability. Plus, it's often on sale for under $10.

        [3] -> [2]
        I have never heard of Stardew Valley. Can I play it on my XBox? What is Steam btw?
        </example>
    """
    return textwrap.dedent(s)


def _get_user_selection_prompt(system: System, thread: Thread) -> str:
    """Select a user posting next"""
    users_str = "\n\n".join(format_user(x) for x in system.users)
    thread_str = format_thread(thread)

    s = f"""
        You're given an ongoing discussion thread, and your task is to select a user who is going to post next to the thread.
        {system.gamemaster.strategy}

        Here is the pool of usernames. Select one from the list, or just answer RANDOM if want to throw a dice.
        <users>
        {users_str}
        </users>


        Here is the ongoing thread.
        <thread>
        {thread_str}
        </thread>

        From the discussion and nature of participants, Who would be the one posting next to this thread? Please answer using the format like this:

        <example>
        User: dont_use_this_sample
        </example>
        """
    return textwrap.dedent(s)


def _get_thread_opening_prompt(instruction: str) -> str:
    """Generate an OP comment creating a thread"""
    s = f"""
        Create a short message to open a thread as Original Poster (OP). Here is the topic of the thread:
        {instruction}

        <example>
        Hey fellow patient gamers! I'm looking for some cheap and fun games on Steam that are worth the wait. Any recommendations?
        </example>
    """
    return textwrap.dedent(s)


def format_post(post: Post) -> str:
    """Format the post as a string."""
    if post.in_reply_to:
        numbers = ",".join(str(i) for i in post.in_reply_to)
        reply = f" -> [{numbers}]"
    else:
        reply = ""
    return f"[{post.id}]{reply}\n{post.text}"


def format_thread(thread: Thread) -> str:
    """Format the thread as a string."""
    return "\n\n".join(format_post(post) for post in thread.posts)


def format_user(user: User) -> str:
    """Format user for user-selection prompt"""
    s = f"""
        - **{user.character}**
            - **Role in Discussion:** {user.role}
    """
    return textwrap.dedent(s)


def init_thread(system: System, instruction: str) -> Thread:
    """Create a thread"""
    prompt = _get_thread_opening_prompt(instruction)
    text = system.gamemaster.generate(prompt)
    post = Post(id=1, username="OP", text=text)
    thread = Thread(id=1, theme=instruction, posts=[post])
    return thread


def update_thread(system: System, thread: Thread) -> None:
    """Extend thread"""
    user = select_user(system, thread)
    post = gen_post(user, thread)
    thread.posts.append(post)


def main() -> None:
    """Main function"""
    logging.basicConfig(level=logging.INFO)
    users = load_users()
    gamemaster = load_gamemaster()
    system = System(gamemaster=gamemaster, users=users)

    instruction = "The topic is about recommendations on cheap and fun games on Steam."
    thread = init_thread(system, instruction)
    for _ in range(5):
        update_thread(system, thread)

    print(format_thread(thread))

