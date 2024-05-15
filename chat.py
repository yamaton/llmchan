"""
Chat handler

"""

import json
import logging
import os
import pathlib
import random
import re
import textwrap
from typing import Literal, Tuple

from pydantic import BaseModel, Field
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam as OpenAIMessageParam
from anthropic import Anthropic
from anthropic.types import MessageParam as AnthropicMessageParam
from anthropic.types import Message as AnthropicMessage
from dotenv import load_dotenv

load_dotenv()

Model = Literal[
    "gpt-4-turbo-2024-04-09",
    "gpt-4o-2024-05-13",
    "claude-3-opus-20240229",
]

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)
if OPENAI_API_KEY is None:
    raise ValueError("Failed to get env $OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", None)
if ANTHROPIC_API_KEY is None:
    raise ValueError("Failed to get env $ANTHROPIC_API_KEY")

TMP_MODEL: Model = "gpt-4o-2024-05-13"


class Agent(BaseModel):
    """OpenAI LLM Agent"""

    model: Model = Field(TMP_MODEL, description="OpenAI LLM model name")

    def generate(
        self, prompt: str, temperature: float, prefill: str | None = None
    ) -> str:
        """Generate a response based on the prompt."""
        client = OpenAI(api_key=OPENAI_API_KEY)

        messages: list[OpenAIMessageParam] = [{"role": "user", "content": prompt}]
        logging.info("[LLM] Generating response from OpenAI LLM")

        chat_completion = client.chat.completions.create(
            messages=messages,
            model=self.model,
            max_tokens=1024,
            temperature=temperature,
        )

        res = chat_completion.choices[0].message.content
        if res is None:
            raise ValueError("Failed to get a response from OpenAI LLM")
        return res


class AnthropicAgent(BaseModel):
    """Anthropic LLM agent"""

    model: Model = Field(TMP_MODEL, description="OpenAI LLM model name")

    def generate(
        self, prompt: str, temperature: float, prefill: str | None = None
    ) -> str:
        """Generate a response based on the prompt.

        prefill: https://docs.anthropic.com/en/docs/prefill-claudes-response

        """
        client = Anthropic(api_key=ANTHROPIC_API_KEY)
        messages: list[AnthropicMessageParam] = [
            {
                "role": "user",
                "content": prompt,
            }
        ]
        if prefill:
            response_message: AnthropicMessageParam = {
                "role": "assistant",
                "content": prefill,
            }
            messages.append(response_message)

        logging.info("[LLM] Generating response from Anthropic LLM")
        response: AnthropicMessage = client.messages.create(
            messages=messages,
            model=self.model,
            max_tokens=1024,
            temperature=temperature,
        )
        return response.content[0].text


class User(BaseModel):
    """User as a chart participant."""

    character: str
    role: str
    agent: Agent

    def generate(
        self, prompt: str, temperature: float = 1.0, prefill: str | None = None
    ) -> str:
        """Get a LLM response"""
        return self.agent.generate(prompt, temperature, prefill)


class GameMaster(BaseModel):
    """Game master controlling who to post next."""

    strategy: str
    agent: Agent

    def generate(
        self, prompt: str, temperature: float = 0.1, prefill: str | None = None
    ) -> str:
        """Get a LLM response"""
        return self.agent.generate(prompt, temperature, prefill)


class System(BaseModel):
    """Chat system"""

    gamemaster: GameMaster
    users: list[User]


class Post(BaseModel):
    """Post in a thread."""

    id: int
    username: str
    text: str
    in_reply_to: list[int] = []


class Thread(BaseModel):
    """A discussion thread"""

    id: int
    topic: str
    posts: list[Post]


def load_users() -> list[User]:
    """Load users from the JSON data file."""
    p = pathlib.Path("data/users.json")
    with p.open("r", encoding="utf8") as f:
        users = []
        for userdata in json.load(f):
            agent = Agent(model=TMP_MODEL)
            user = User(**userdata, agent=agent)
            users.append(user)
    return users


def load_gamemaster() -> GameMaster:
    """Load the game master from the JSON data file."""
    p = pathlib.Path("data/strategies.json")
    with p.open("r", encoding="utf8") as f:
        agent = Agent(model=TMP_MODEL)  # hardcoded for now
        data_list = json.load(f)

    userdata = data_list[0]  # [TODO] Hardcoded. Change later.
    gamemaster = GameMaster(**userdata, agent=agent)
    return gamemaster


def select_user(system: System, thread: Thread) -> User:
    """Select a user posting next"""
    prompt = _get_user_selection_prompt(system=system, thread=thread)
    logging.debug(f"Prompt to select a user: {prompt}")
    logging.info("Selecting a user to post next")
    response = system.gamemaster.generate(prompt, prefill="User: ")
    if not response or not response.startswith("User: "):
        raise ValueError(f"LLM behaving weird: {response}")

    r = response.lower()
    for user in system.users:
        if user.character.lower() in r:
            selected = user
            break
    else:
        selected = random.choice(system.users)
        if "random" in r:
            logging.info(f"LLM response: {response}")
        else:
            logging.warning(f"[Fallback] LLM response: {r}")

    logging.info(f"Chosen user: {selected.character}")
    return selected


def gen_post(user: User, thread: Thread) -> Post:
    """Generate a post for the user."""
    id_ = thread.posts[-1].id + 1 if thread.posts else 1
    prompt = _get_user_prompt(user, thread)
    logging.debug(f"Prompt for post generation: {prompt}")
    logging.info(f"Generating post for {user.character}")
    text = user.generate(prompt)
    if text is None:
        raise ValueError("Failed to generate text.")

    in_reply_to = _extract_reply_to(text)
    text = _clean_text(text)
    return Post(id=id_, username=user.character, text=text, in_reply_to=in_reply_to)


def _extract_reply_to(text: str) -> list[int]:
    """Extract the reply-to post IDs from the text."""
    if " -> " in text:
        for line in text.split("\n"):
            if " -> " in line:
                reply_to_part = line.split(" -> ")[1].strip()[1:-1]
                return [int(x) for x in reply_to_part.split(",")]
    return []


def _clean_text(text: str) -> str:
    """Clean up the text"""
    lines = text.strip().split("\n")
    return "\n".join(
        x
        for x in lines
        if (
            (not x.startswith("["))
            and (not x.startswith("<example>"))
            and (not x.startswith("</example>"))
            and (not x.startswith("(*)"))
        )
    ).strip()


def _get_user_prompt(user: User, thread: Thread) -> str:
    """Generate a prompt for user content generation"""
    s = f"""\
        ### Objective

        Post a single message to this given discussion thread:

        <thread>
        {textwrap.indent(format_thread_user(thread, user), " " * 8)}
        </thread>

        ### Role
        {textwrap.indent(user.role, " " * 8)}

        ### Thread format

        * Participants are anonymous and no usernames is displayed, but your own posts are marked with an asterisk `(*)`.
        * You may use "->" to indicate replies, but omit reply to the thread opener. In other words, never use `-> [0]`. Use the format ` -> [3,5]` to reply to multiple posts, but never reply to more than two.
        * Keep a comment 1 to 5 senstences long.
        * The discussion is in the form of temporally-ordered messages as follows. (`<example>` and `</example>` are not part of the message.)

        <example>
        [0]
        Hey fellow patient gamers! I'm looking for some cheap and fun games on Steam that are worth the wait. Any recommendations?

        [1]
        Stardew Valley is a must-play! It's a charming farming RPG with tons of content and replayability. Plus, it's often on sale for under $10.

        [2] -> [1]
        I have never heard of Stardew Valley. Can I play it on my XBox? What is Steam btw?
        </example>

        ### Message format
        Your message should be in the following format. `<example>` and `</example>` are not part of the message.

        <example>
        [3] -> [2]
        I'm not sure about XBox, but you can definitely play Stardew Valley on Steam. Steam is a digital distribution platform for video games.
        </example>
    """
    return textwrap.dedent(s)


def _get_user_selection_prompt(system: System, thread: Thread) -> str:
    """Select a user posting next"""
    users_str = "\n\n".join(format_user(x) for x in system.users)
    thread_str = format_thread(thread)

    s = f"""\
        You're given an ongoing discussion thread, and your task is to select a user who is going to post next to the thread.
        Here is your stategy in choosing the next user:
        {textwrap.indent(system.gamemaster.strategy, " " * 8)}

        Here is the pool of usernames from which you are to select. You may also answer RANDOM if casting a dice.
        <users>
        {textwrap.indent(users_str, " " * 8)}
        </users>

        Here is the ongoing thread.
        <thread>
        {textwrap.indent(thread_str, " " * 8)}
        </thread>

        Based on these information, who would be the one posting next to this thread?
        Please answer using the format like this:

        User: dont_use_this_sample
        """
    return textwrap.dedent(s)


def _get_thread_opening_prompt(topic: str) -> str:
    """Generate an OP comment creating a thread"""
    s = f"""\
        ### Objective

        Create a short message to open a thread as Original Poster (OP). Here is the topic of the thread:
        {textwrap.indent(topic, " " * 8)}

        ### Message format
        Your message should be in the following format. `<example>` and `</example>` are not part of the message.

        <example>
        Hey fellow patient gamers! I'm looking for some cheap and fun games on Steam that are worth the wait. Any recommendations?
        </example>
    """
    return textwrap.dedent(s)


def format_post(post: Post, user: User) -> str:
    """Format the post as a string for the `user`."""
    mypost = "(*) " if post.username == user.character else ""

    if post.in_reply_to:
        numbers = ",".join(str(i) for i in post.in_reply_to)
        reply = f" -> [{numbers}]"
    else:
        reply = ""
    return f"{mypost}[{post.id}]{reply}\n{post.text}"


def format_post_with_username(post: Post) -> str:
    """Format the post as a string."""
    if post.in_reply_to:
        numbers = ",".join(str(i) for i in post.in_reply_to)
        reply = f" -> [{numbers}]"
    else:
        reply = ""
    return f"[{post.id}] {post.username}{reply}\n{post.text}"


def format_thread(thread: Thread) -> str:
    """Format the thread as a string. Includes usernames."""
    return "\n\n".join(format_post_with_username(post) for post in thread.posts)


def format_thread_user(thread: Thread, user: User) -> str:
    """Format the thread as a string for the `user` to read.

    - The user's own posts are marked with an asterisk.
    - Usernames are not displayed.
    """
    return "\n\n".join(format_post(post, user) for post in thread.posts)


def format_user(user: User) -> str:
    """Format user for user-selection prompt"""
    s = f"""- **{user.character}**\n    - **Role in Discussion:** {user.role}"""
    return textwrap.dedent(s)


def _parse_thread_header(line: str) -> Tuple[int, str, list[int]]:
    """Parse the thread header line, and extract
    - thread ID
    - username
    - reply-to post IDs

    Example:
    [0] OP -> [1,2]
    """
    parts = line.split(" -> ")
    chunk = parts[0]
    id_and_username = chunk.split("]")
    id_ = int(id_and_username[0][1:])
    username = id_and_username[1].strip()
    reply_to = []
    if len(parts) > 1:
        xs = parts[1].strip()[1:-1].split(",")
        reply_to = [int(x) for x in xs]
    return id_, username, reply_to


def _parse_single_post(text: str) -> Post:
    """Parse a single post."""
    lines = text.strip().split("\n")
    id_, username, reply_to = _parse_thread_header(lines[0])
    text = "\n".join(lines[1:])
    return Post(id=id_, username=username, text=text, in_reply_to=reply_to)


def load_thread(text: str) -> Thread:
    """Load a thread by parsing the text."""
    patt = re.compile("\n\n(?=\\[)")
    chunks = patt.split(text.strip())
    posts = [_parse_single_post(chunk) for chunk in chunks]
    return Thread(id=0, topic="", posts=posts)


def init_thread(system: System, topic: str) -> Thread:
    """Create a thread"""
    logging.info("Creating a new thread")
    prompt = _get_thread_opening_prompt(topic)
    logging.debug(f"Prompt to start a thread: {prompt}")
    text = system.gamemaster.generate(prompt, temperature=0.8)
    text = _clean_text(text)
    post = Post(id=0, username="OP", text=text)
    thread = Thread(id=0, topic=topic, posts=[post])
    return thread


def update_thread(system: System, thread: Thread) -> None:
    """Extend thread"""
    logging.info("Updating the thread")
    user = select_user(system, thread)
    post = gen_post(user, thread)
    thread.posts.append(post)


def main() -> None:
    """Main function"""
    logging.basicConfig(level=logging.INFO)
    users = load_users()
    gamemaster = load_gamemaster()
    system = System(gamemaster=gamemaster, users=users)

    # topic = "Recommendations on fun and cheap games on Steam."
    # topic = "Tabby's Star, a mysterious star showing irregularly fluctuating luminosity."
    # topic = "Why does American government keep sending huge chunk of money to Israel?"
    # topic = "Alternatives to Nvidia's CUDA in AI computation."
    # topic = "Issues in American political campaign financing, and how to fix it."
    # topic = "How does nature maintain biodiversity by overcoming the competitive exclusion principle?"
    # topic = "How does the current AI hype end up in a bubble burst? Or, does it?"
    # topic = "What happened to the Metaverse and VR/AR hype in the recent years?"
    # topic = "源氏物語の宇治十帖について日本語で語り合いましょう。"
    topic = "How can we understand that 1 + 2 + 3 + ... = -1/12?"
    # topic = textwrap.dedent("""
    # Suppose that $a$, $b$, $c$, $d$ are positive real numbers satisfying $(a + c)(b + d) = ac + bd$.
    # Find the smallest possible value of
    # $$
    # \frac{a}{b} + \frac{b}{c} + \frac{c}{d} + \frac{d}{a}.
    # $$
    # """).strip()

    thread = init_thread(system, topic)
    print(">>>---------------------------------------")
    print(format_thread(thread))
    print("<<<---------------------------------------")

    for _ in range(5):
        update_thread(system, thread)
        print(">>>---------------------------------------")
        print(format_thread(thread))
        print("<<<---------------------------------------")

    print(">>>=======================================")
    print(format_thread(thread))
    print("<<<=======================================")


if __name__ == "__main__":
    main()
