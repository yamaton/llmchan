import logging
from pathlib import Path
from textual import on
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, TextArea, RichLog, Button, Static, Select
from textual.containers import Horizontal, Vertical

import chat

TOPIC = chat.TOPICS[1]

logging.basicConfig(
    level=logging.INFO,
    filename="myapp.log",
)

PATH_BASE = Path("chanlog")
Languages: list[chat.Language] = [
    "en", "ja", "es", "ru", "fr", "de", "it", "pt", "ko", "zh", "ar", "hi", "tr", "vi", "th", "id"
]


class Chan(App):

    TITLE = "llmchan"
    SUB_TITLE = "A discussion board"
    CSS_PATH = "llmchan.tcss"

    BINDINGS = [
        ("escape", "toggle_comment", "Toggle Input Area"),
        ("space", "load_post", "Load next post"),
        ("ctrl+s", "submit_text", "Submit text"),
        ("i", "toggle_settings", "Toggle Settings"),
    ]

    system = chat.init_system()
    saveid = chat.gen_unique_id()
    thread: chat.Thread | None = None
    lang: chat.Language = "en"

    def compose(self) -> ComposeResult:
        """Compose the layout of the app."""
        yield Header(name="llmchan")
        yield Select(
            ((line, line) for line in chat.TOPICS),
            prompt="Select a topic",
            id="select_topic",
        )
        yield Select(
            ((lang, lang) for lang in Languages),
            prompt="Select a language",
            id="select_lang",
        )
        yield RichLog(wrap=True, id="rich_log")
        with Horizontal(id="userinput"):
            yield TextArea(id="textarea")
            yield Button("Submit", id="submit_button", variant="default")
        yield Footer()

    def on_ready(self) -> None:
        """Called when the app is ready."""
        rich_log = self.query_one(RichLog)
        if self.thread:
            s = chat.format_thread(self.thread)
        else:
            s = "(No thread loaded.)"
        rich_log.write(s)

        # Configure Widgets
        select_topic: Select = self.query_one("#select_topic")  # type: ignore
        select_topic.expanded = True
        select_lang = self.query_one("#select_lang")
        select_lang.add_class("hidden")


    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Event handler called when a button is pressed."""
        button_id = event.button.id
        if button_id == "submit_button":
            self.action_submit_text()

    def action_toggle_comment(self) -> None:
        """Action to toggle the user-input area."""
        logging.info("Called: action_toggle_comment")
        comment = self.query_one(Horizontal)
        if comment.has_class("hidden"):
            comment.remove_class("hidden")
        else:
            comment.add_class("hidden")

    def action_submit_text(self) -> None:
        """Action to submit text from the text area."""
        text_area = self.query_one(TextArea)
        rich_log = self.query_one(RichLog)
        comment = text_area.text.strip()
        if comment and self.thread:
            text_area.text = ""
            post = chat.add_message(self.thread, comment)
            s = chat.format_post_with_username(post)
            rich_log.write("\n\n" + s)
            logging.info("Comment: %s", comment)

    def action_load_post(self) -> None:
        """Action to load the next post."""
        logging.info("Called: action_load_post")
        if self.thread:
            rich_log = self.query_one(RichLog)
            post = chat.update_thread(self.system, self.thread)
            s = chat.format_post_with_username(post)
            rich_log.write("\n\n" + s)
            p = Path(f"{PATH_BASE}_{self.thread.topic[:10]}_{self.saveid}.txt")
            chat.save_thread(p, self.thread)

    def action_toggle_settings(self) -> None:
        """Action to toggle the settings."""
        logging.info("Called: action_toggle_settings")
        select_lang = self.query_one("#select_lang")
        if select_lang.has_class("hidden"):
            select_lang.remove_class("hidden")
        else:
            select_lang.add_class("hidden")


    def select_topic(self, topic: str) -> None:
        """Select a topic."""
        self.thread = chat.init_thread(self.system, topic)

        # Initialize the thread display
        rich_log = self.query_one(RichLog)
        rich_log.clear()
        s = chat.format_thread(self.thread)
        rich_log.write(s)

        # Hide the topic-selection widget
        select_topic = self.query_one("#select_topic")
        select_topic.add_class("hidden")


    def select_lang(self, lang: chat.Language) -> None:
        """Select a language."""
        self.lang = lang
        select_lang = self.query_one("#select_lang")
        for user in self.system.users:
            user.set_lang(lang)

        select_lang.add_class("hidden")
        logging.info("Language: %s", lang)


    @on(Select.Changed)
    def select_changed(self, event: Select.Changed) -> None:
        """Event handler called when the select widget changes."""
        logging.info("Called: select_changed")
        if event.select.id == "select_lang":
            self.select_lang(str(event.value))  # type: ignore
        elif event.select.id == "select_topic":
            self.select_topic(str(event.value))


if __name__ == "__main__":
    app = Chan()
    app.run()
