import logging
from pathlib import Path
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, TextArea, RichLog, Button, Static
from textual.containers import Horizontal, Vertical

import chat

TOPIC = chat.TOPICS[1]

logging.basicConfig(
    level=logging.INFO,
    filename="myapp.log",
)

PATH_BASE = Path("chanlog")


class Chan(App):

    TITLE = "llmchan"
    SUB_TITLE = "A discussion board"
    CSS_PATH = "llmchan.tcss"

    BINDINGS = [
        ("escape", "toggle_comment", "Toggle Input Area"),
        ("space", "load_post", "Load next post"),
        ("ctrl+s", "submit_text", "Submit text"),
    ]

    system = chat.init_system()
    thread = chat.init_thread(system, TOPIC)
    saveid = chat.gen_unique_id()

    def on_ready(self) -> None:
        """Called when the app is ready."""
        rich_log = self.query_one(RichLog)
        s = chat.format_thread(self.thread)
        rich_log.write(s)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Event handler called when a button is pressed."""
        button_id = event.button.id
        if button_id == "submit_button":
            self.action_submit_text()

    def compose(self) -> ComposeResult:
        """Compose the layout of the app."""
        yield Header(name="llmchan")
        yield RichLog(wrap=True, id="rich_log")
        with Horizontal(id="userinput"):
            yield TextArea(id="textarea")
            yield Button("Submit", id="submit_button", variant="default")
        yield Footer()

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
        if comment:
            text_area.text = ""
            post = chat.add_message(self.thread, comment)
            s = chat.format_post_with_username(post)
            rich_log.write("\n\n" + s)
            logging.info("Comment: %s", comment)

    def action_load_post(self) -> None:
        """Action to load the next post."""
        logging.info("Called: action_load_post")
        rich_log = self.query_one(RichLog)
        post = chat.update_thread(self.system, self.thread)
        s = chat.format_post_with_username(post)
        rich_log.write("\n\n" + s)
        p = Path(f"{PATH_BASE}_{self.thread.topic[:10]}_{self.saveid}.txt")
        chat.save_thread(p, self.thread)


if __name__ == "__main__":
    app = Chan()
    app.run()
