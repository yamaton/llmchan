import logging
from pathlib import Path
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, TextArea, RichLog, Button
from textual.containers import Horizontal

import chat

# topic = "How can we understand that 1 + 2 + 3 + ... = -1/12?"
topic = "マイナーだけど最高に面白いマンガについて語ろう。"


logging.basicConfig(
    level=logging.INFO,
    filename="myapp.log",
)

PATH_THREAD = Path("thread.txt")


class Chan(App):

    CSS = """
    Horizontal {
        height: 10;
    }

    Button {
        margin: 1;
    }

    TextArea {
        margin: 1 1;
    }

    .hidden {
        display: none;
    }
    """

    BINDINGS = [
        ("escape", "toggle_comment", "Toggle Input Area"),
        ("space", "load_post", "Load next post"),
        ("ctrl+s", "submit_text", "Submit text"),
    ]

    system = chat.init_system()
    thread = chat.init_thread(system, topic)


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
        chat.save_thread(PATH_THREAD, self.thread)


if __name__ == "__main__":
    app = Chan()
    app.run()
