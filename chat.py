"""
Chat handler

"""

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

load_dotenv()

Model = Literal[
    "gpt-4-turbo-2024-04-09",
    "gpt-4o-2024-05-13",
    "claude-3-opus-20240229",
]

Language = Literal[
    "en",
    "ja",
    "es",
    "ru",
    "fr",
    "de",
    "it",
    "pt",
    "ko",
    "zh",
    "ar",
    "hi",
    "tr",
    "vi",
    "th",
    "id",
]


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)
if OPENAI_API_KEY is None:
    raise ValueError("Failed to get env $OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", None)
if ANTHROPIC_API_KEY is None:
    raise ValueError("Failed to get env $ANTHROPIC_API_KEY")

TMP_MODEL: Model = "gpt-4o-2024-05-13"


# [TODO] Move to a separate file
# For multi-lingual support


LangPromptDict: dict[Language, str] = {
    "en": "",
    "ja": "Use Japanese language only; 日本語で答えてください。",
    "es": "Use Spanish language only; Responde en español, por favor.",
    "ru": "Use Russian language only; Ответьте на русском языке, пожалуйста.",
    "fr": "Use French language only; Répondez en français, s'il vous plaît.",
    "de": "Use German language only; Bitte antworten Sie auf Deutsch.",
    "it": "Use Italian language only; Rispondi in italiano, per favore.",
    "pt": "Use Portuguese language only; Responda em português, por favor.",
    "ko": "Use Korean language only; 한국어로 답변해주세요.",
    "zh": "Use Chinese language only; 请用中文回答。",
    "ar": "Use Arabic language only; أجب باللغة العربية من فضلك.",
    "hi": "Use Hindi language only; कृपया हिंदी में जवाब दें।",
    "tr": "Use Turkish language only; Lütfen Türkçe cevap verin.",
    "vi": "Use Vietnamese language only; Trả lời bằng tiếng Việt, xin cảm ơn.",
    "th": "Use Thai language only; โปรดตอบด้วยภาษาไทย",
    "id": "Use Indonesian language only; Tolong jawab dalam bahasa Indonesia.",
}


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
            raise ValueError("Failed to get a response from OpenAI LLM")
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

    lang: Language = Field("en", description="Language to use for the response")

    def generate(self, prompt: str, temperature: float, prefill: None | str = "") -> str:
        """Generate a response based on the prompt."""
        system_prompt = LangPromptDict[self.lang]
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
    agent: Agent

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
    p = Path("data/users.json")
    with p.open("r", encoding="utf8") as f:
        users = []
        for userdata in json.load(f):
            if "lang" in userdata:
                agent = LangAgent(model=TMP_MODEL, lang=userdata["lang"])
            else:
                agent = LangAgent(model=TMP_MODEL, lang="en")
            user = User(**userdata, agent=agent)
            users.append(user)
    return users


def load_gamemaster() -> GameMaster:
    """Load the game master from the JSON data file."""
    p = Path("data/strategies.json")
    with p.open("r", encoding="utf8") as f:
        agent = Agent(model=TMP_MODEL)  # hardcoded for now
        data_list = json.load(f)

    userdata = data_list[0]  # [TODO] Hardcoded. Change later.
    gamemaster = GameMaster(**userdata, agent=agent)
    return gamemaster


def init_system() -> System:
    """Initialize the chat system"""
    gamemaster = load_gamemaster()
    users = load_users()
    return System(gamemaster=gamemaster, users=users)


def select_user(system: System, thread: Thread) -> User:
    """Select a user posting next"""
    prompt = _get_user_selection_prompt(system=system, thread=thread)
    logging.info("[select_user]")
    logging.debug(f"[select_user] prompt: {prompt}")
    response = system.gamemaster.generate(prompt, prefill="User: ")
    if not response or not response.startswith("User: "):
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
    """Generate a post for the user."""
    id_ = thread.posts[-1].id + 1 if thread.posts else 1
    prompt = _get_user_prompt(user, thread)
    logging.info(f"[gen_post] for {user.character}")
    logging.debug(f"[gen_post] prompt:\n{prompt}")
    text = user.generate(prompt, temperature=0.9)
    logging.debug(f"[gen_post] response:\n{text}")
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
    logging.debug(f"[init_thread] response:\n{text}")
    text = _clean_text(text)
    logging.debug(f"[init_thread] response after cleaning: \n{text}")
    post = Post(id=0, username="OP", text=text)
    thread = Thread(id=0, topic=topic, posts=[post])
    return thread


def update_thread(system: System, thread: Thread) -> Post:
    """Extend thread"""
    logging.info("[update_thread] selecting a user")
    user = select_user(system, thread)
    logging.info("[update_thread] generating a post")
    post = gen_post(user, thread)
    thread.posts.append(post)
    return post


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
    "Recommendations on fun and cheap games on Steam.",
    "Tabby's Star, a mysterious star showing irregularly fluctuating luminosity.",
    "Alternatives to Nvidia's CUDA in AI computation.",
    "Issues in American political campaign financing, and how to fix it.",
    "How does nature maintain biodiversity by overcoming the competitive exclusion principle?",
    "How does the current AI hype end up in a bubble burst? Or, does it?",
    "What happened to the Metaverse and VR/AR hype in the recent years?",
    "源氏物語の宇治十帖について日本語で語り合いましょう。",
    "マイナーだけど最高に面白いマンガについて語ろう。",
    "What should aging societies like Japan and China do to maintain their economy?",
    "How can we understand that 1 + 2 + 3 + ... = -1/12?",
    textwrap.dedent("""
    Suppose that $a$, $b$, $c$, $d$ are positive real numbers satisfying $(a + c)(b + d) = ac + bd$.
    Find the smallest possible value of
    $$
    \frac{a}{b} + \frac{b}{c} + \frac{c}{d} + \frac{d}{a}.
    $$
    """).strip(),
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
