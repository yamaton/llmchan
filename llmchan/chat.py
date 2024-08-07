"""
Chat handler

"""

import asyncio
import concurrent.futures
import base64
import json
import logging
import os
import random
import re
import textwrap
import uuid
from pathlib import Path
from typing import Literal, Tuple

from pydantic import BaseModel, Field
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam as OpenAIMessageParam
from anthropic import Anthropic
from anthropic.types import MessageParam as AnthropicMessageParam
from anthropic.types import Message as AnthropicMessage
from dotenv import load_dotenv

from .localization import Language, LangPromptDict

load_dotenv()

Model = Literal[
    "gpt-4-turbo-2024-04-09",
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
    "gpt-4o-mini-2024-07-18",
    "claude-3-opus-20240229",
]


TMP_MODEL: Model = "gpt-4o-mini-2024-07-18"


class APIKeyError(Exception):
    """Raised when the API key is not set."""

    pass


class Agent(BaseModel):
    """OpenAI LLM Agent"""

    model: Model = Field(TMP_MODEL, description="OpenAI LLM model name")

    def generate(
        self,
        prompt: str,
        temperature: float,
        system_prompt: str,
        prefill: str | None = None,
    ) -> str:
        """Generate a response based on the prompt."""
        OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)
        if OPENAI_API_KEY is None:
            logging.error(
                "[APIKeyError] Please set the environment variable OPENAI_API_KEY."
            )
            raise APIKeyError("Please set the environment variable OPENAI_API_KEY.")

        client = OpenAI(api_key=OPENAI_API_KEY)

        messages: list[OpenAIMessageParam] = [{"role": "user", "content": prompt}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        logging.info("[LLM] Generating response from OpenAI LLM")

        chat_completion = client.chat.completions.create(
            messages=messages,
            model=self.model,
            max_tokens=1024,
            temperature=temperature,
        )

        res = chat_completion.choices[0].message.content
        if res is None:
            logging.error(
                "[Agent.generate] OpenAI chat completion came with an empty text..."
            )
            raise ValueError("Failed to get a response from OpenAI.")
        return res


class AnthropicAgent(BaseModel):
    """Anthropic LLM agent"""

    model: Model = Field(TMP_MODEL, description="OpenAI LLM model name")

    def generate(
        self,
        prompt: str,
        temperature: float,
        system_prompt: str,
        prefill: str | None = None,
    ) -> str:
        """Generate a response based on the prompt.

        prefill: https://docs.anthropic.com/en/docs/prefill-claudes-response

        """
        ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", None)
        if ANTHROPIC_API_KEY is None:
            logging.error(
                "[AnthropicAgent.generate] Please set the environment variable ANTHROPIC_API_KEY."
            )
            raise APIKeyError("Please set the environment variable ANTHROPIC_API_KEY.")

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
        kwargs = {
            "messages": messages,
            "model": self.model,
            "max_tokens": 1024,
            "temperature": temperature,
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        response: AnthropicMessage = client.messages.create(**kwargs)
        return response.content[0].text


class LangAgent(Agent):

    lang: Language = Field("english", description="Language to use for the response")

    def generate(
        self,
        prompt: str,
        temperature: float,
        system_prompt: str = "",
        prefill: None | str = "",
    ) -> str:
        """Generate a response based on the prompt."""
        system_prompt = "\n\n".join([system_prompt, LangPromptDict[self.lang]]).strip()
        return super().generate(prompt, temperature, system_prompt, prefill)

    def set_lang(self, language: Language):
        """Set the language of the agent."""
        self.lang = language


class User(BaseModel):
    """User as a chart participant."""

    character: str
    role: str
    agent: LangAgent

    def generate(
        self,
        prompt: str,
        temperature: float = 1.0,
        prefill: str | None = None,
    ) -> str:
        """Get a LLM response"""
        return self.agent.generate(prompt, temperature, prefill=prefill)

    def set_lang(self, language: Language):
        """Set the language of the agent."""
        self.agent.set_lang(language)


class GameMaster(BaseModel):
    """Game master controlling who to post next."""

    strategy: str
    agent: LangAgent

    def generate(
        self,
        prompt: str,
        temperature: float = 0.1,
        system_prompt: str = "",
        prefill: str | None = None,
    ) -> str:
        """Get a LLM response"""
        return self.agent.generate(
            prompt, temperature, system_prompt=system_prompt, prefill=prefill
        )

    def set_lang(self, language: Language):
        """Set the language of the agent."""
        self.agent.set_lang(language)


class System(BaseModel):
    """Chat system"""

    gamemaster: GameMaster
    users: list[User]

    def set_lang(self, language: Language):
        """Set the language of the agents."""
        self.gamemaster.set_lang(language)
        for user in self.users:
            user.set_lang(language)


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
    p = Path(__file__).parent / "data" / "users.json"
    with p.open("r", encoding="utf8") as f:
        users = []
        for userdata in json.load(f):
            if "lang" in userdata:
                agent = LangAgent(model=TMP_MODEL, lang=userdata["lang"])
            else:
                agent = LangAgent(model=TMP_MODEL, lang="english")
            user = User(**userdata, agent=agent)
            users.append(user)
    return users


def load_gamemaster() -> GameMaster:
    """Load the game master from the JSON data file."""
    p = Path(__file__).parent / "data" / "strategies.json"
    with p.open("r", encoding="utf8") as f:
        agent = LangAgent(model=TMP_MODEL, lang="english")  # hardcoded for now
        data_list = json.load(f)

    userdata = data_list[-1]  # [TODO] Hardcoded. Change later.
    gamemaster = GameMaster(**userdata, agent=agent)
    return gamemaster


def init_system() -> System:
    """Initialize the chat system"""
    gamemaster = load_gamemaster()
    users = load_users()
    return System(gamemaster=gamemaster, users=users)


def select_user(system: System, thread: Thread) -> User:
    """Select a user posting next

    NOTE: Require a LLM inquiery.
    """
    prompt = _get_user_selection_prompt(system=system, thread=thread)
    logging.info("[select_user]")
    logging.debug(f"[select_user] prompt: {prompt}")
    response = system.gamemaster.generate(prompt, prefill="User: ")
    if not response or not response.startswith("User: "):
        logging.error(f"[select_user] LLM behaving weird: {response}.")
        raise ValueError(f"LLM behaving weird: {response}")
    logging.debug(f"[select_user] response:\n{response}")

    r = response.lower()
    for user in system.users:
        if user.character.lower() in r:
            selected = user
            break
    else:
        selected = random.choice(system.users)
        if "random" in r:
            logging.info(f"[select_user] choosing random user.")
        else:
            logging.warning(f"[select_user] [FALLBACK] LLM response:\n{r}")

    logging.info(f"[select_user] choice: {selected.character}")
    return selected


def gen_post(user: User, thread: Thread) -> Post:
    """Generate a post for the user.

    NOTE: Require a LLM inquiery.
    """
    id_ = thread.posts[-1].id + 1 if thread.posts else 1
    prompt = _get_user_prompt(user, thread)
    logging.info(f"[gen_post] for {user.character}")
    logging.debug(f"[gen_post] prompt:\n{prompt}")
    text = user.generate(prompt, temperature=0.9)
    logging.debug(f"[gen_post] response:\n{text}")
    if not text:
        logging.error("[gen_post] Failed to generate text.")
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
    """Prepare a prompt for user content generation"""
    s = f"""\
        ### Objective

        Post a single message to this given discussion thread:

        <thread>
        {textwrap.indent(format_thread_user(thread, user), " " * 8)}
        </thread>

        ### Role
        {textwrap.indent(user.role, " " * 8)}
        Use the tone and style that fits your role.

        ### Thread format

        * Participants are anonymous and no usernames is displayed, but your own posts are marked with an asterisk `(*)`.
        * You may use "->" to indicate replies **only if** it's really necessary. Alway omit reply to the thread opener: never use `-> [0]`. Use the format ` -> [3,5]` to reply to two posts. Never reply to more than two.
        * Keep a comment 1 to 5 senstences long.
        * The discussion is in the form of temporally-ordered messages as follows. (`<example>` and `</example>` are not part of the message.)

        <example>
        [0]
        Hey fellow patient gamers! I'm looking for some cheap and fun games on Steam that are worth the wait. Any recommendations?

        [1]
        Stardew Valley is a must-play! It's a charming farming RPG with tons of content and replayability. Plus, it's often on sale for under $10.

        [2]
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
    """Prepare a user-posting prompt"""
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
    """Prepare a prompt for OP post."""
    s = f"""\
        ### Objective

        Create a short message to open a thread as Original Poster (OP). Here is the topic of the thread:
        {textwrap.indent(topic, " " * 8)}

        ### Message Tone
        Use casual and conversational tone. 

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
    s = f"""- **{user.character}**\n    - **Role in Discussion:** {user.role}\n    - Tone and Style: Use conversational tone, and the style that fits your role."""
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


def parse_as_thread(text: str, id_: int, topic: str) -> Thread:
    """Load a thread by parsing the text."""
    patt = re.compile("\n\n(?=\\[)")
    chunks = patt.split(text.strip())
    posts = [_parse_single_post(chunk) for chunk in chunks]
    return Thread(id=id_, topic=topic, posts=posts)


def init_thread(system: System, topic: str) -> Thread:
    """Create a thread based on the topic."""
    prompt = _get_thread_opening_prompt(topic)
    logging.info("[init_thread] Generating an initial post.")
    logging.debug(f"[init_thread] prompt:\n{prompt}")
    text = system.gamemaster.generate(prompt, temperature=0.9)
    if not text:
        logging.error("[init_thread] Failed to generate the initial-post text.")
        raise ValueError("Failed to generate the initial-post text.")
    logging.debug(f"[init_thread] response:\n{text}")
    text = _clean_text(text)
    logging.debug(f"[init_thread] response after cleaning: \n{text}")
    thread = create_thread_from_text(text, topic)
    return thread


def create_thread_from_text(text: str, topic: str = "__manual__") -> Thread:
    """Create a thread based on the topic."""
    post = Post(id=0, username="OP", text=text.strip())
    thread = Thread(id=0, topic=topic, posts=[post])
    return thread


async def init_thread_async(system: System, topic: str) -> Thread:
    """Async version of init_thread"""
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        result = await loop.run_in_executor(executor, init_thread, system, topic)
    return result


def update_thread(system: System, thread: Thread) -> Post:
    """Extend thread"""
    logging.info("[update_thread] selecting a user")
    user = select_user(system, thread)
    logging.info("[update_thread] generating a post")
    post = gen_post(user, thread)
    thread.posts.append(post)
    return post


async def update_thread_async(system: System, thread: Thread) -> Post:
    """Extend thread"""
    logging.info("[update_thread] selecting a user")
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        result = await loop.run_in_executor(executor, update_thread, system, thread)
    return result


def update_thread_batch(system: System, thread: Thread, n: int) -> None:
    """Extend thread with multiple posts"""
    for _ in range(n):
        update_thread(system, thread)


def save_thread(path: Path, thread: Thread) -> None:
    """Save a thread to a file."""
    with path.open("w", encoding="utf8") as f:
        f.write(format_thread(thread))


def load_thread(path: Path, id_: int = 0, topic: str = "") -> Thread:
    """Load a thread from a file."""
    with path.open("r", encoding="utf8") as f:
        text = f.read()
    return parse_as_thread(text, id_, topic)


def add_message(thread: Thread, message) -> Post:
    """Post a message from a user to the thread"""
    post = Post(id=thread.posts[-1].id + 1, username="Ready_Player_One", text=message)
    thread.posts.append(post)
    return post


def gen_unique_id() -> str:
    """Generate a short unique ID."""
    u = uuid.uuid4()
    s = base64.urlsafe_b64encode(u.bytes).rstrip(b"=").decode("utf-8")
    return s


OTHER_TOPIC = "Other ..."

TOPICS = [
    "Recommendations for Fun and Affordable Games on Steam",
    "Tabby's Star: A Mysterious Star with Irregularly Fluctuating Luminosity",
    "Alternatives to Nvidia's CUDA in AI computation",
    "Problems in American Political Campaign Financing and How to Fix Them",
    "How Nature Maintains Biodiversity Despite the Competitive Exclusion Principle?",
    "Will the Current AI Hype Lead to a Bubble Burst?",
    "What Happened to the Metaverse and VR/AR Hype in Recent Years",
    "Share Your Current Obsession: Embrace the Niche and Unusual",
    "Discussing Lesser-Known but Incredibly Entertaining Manga",
    "Discussing Favorite Science Fiction Novels",
    "Let's Discuss the Uji Chapters of The Tale of Genji",
    "What Should Aging Societies Like Japan and China Do to Maintain Their Economies?",
    "How can we understand 1 + 2 + 3 + ... = -1/12?",
    OTHER_TOPIC,  # this activates the user input box
]


def main() -> None:
    """Main function"""
    topic = random.choice(TOPICS)
    log_handler = logging.FileHandler(f"{topic[:30]}.log")
    logging.basicConfig(level=logging.INFO, handlers=[log_handler])

    system = init_system()
    thread = init_thread(system, topic)
    update_thread_batch(system, thread, 3)
    p = Path("test-thread.txt")
    save_thread(p, thread)


if __name__ == "__main__":
    main()
