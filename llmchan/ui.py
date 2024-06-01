import logging
import datetime
from pathlib import Path

from textual import on, work
from textual.app import App, ComposeResult
from textual.widgets import (
    Header,
    Footer,
    TextArea,
    RichLog,
    Button,
    Select,
    Input,
    LoadingIndicator,
)
from textual.containers import Horizontal, Vertical

from . import chat
from . import localization

nowstr = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    filename=f"llmchan_{nowstr}.log",
)

PATH_BASE = Path("thread")


class Chan(App):

    TITLE = "llmchan"
    SUB_TITLE = "A discussion board"
    CSS_PATH = "llmchan.tcss"

    BINDINGS = [
        ("escape", "toggle_comment", "Toggle Input"),
        ("space", "load_post", "Next"),
        ("ctrl+s", "submit_text", "Submit"),
        ("i", "toggle_settings", "Toggle Settings"),
    ]

    system = chat.init_system()
    saveid = chat.gen_unique_id()
    thread: chat.Thread | None = None
    lang: chat.Language = "en"

    def compose(self) -> ComposeResult:
        """Compose the layout of the app."""
        yield Header(name="llmchan")
        yield LoadingIndicator()
        yield Select.from_values(
            localization.LangList,
            prompt="Select language",
            id="select_lang",
        )
        with Vertical(id="vertical_topic"):
            yield Select.from_values(
                chat.TOPICS,
                prompt="Select a topic",
                id="select_topic",
            )
            yield Input(id="input_topic", placeholder="Enter a topic")
        yield RichLog(wrap=True, id="rich_log")
        with Horizontal(id="userinput"):
            yield TextArea(id="textarea")
            with Vertical(id="buttons"):
                yield Button("Submit", id="submit_button", variant="default")
                yield Button("Next", id="next_button", variant="default")
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
        self._hide("#input_topic")
        self._hide("#select_lang")
        self._hide(LoadingIndicator)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Event handler called when a button is pressed."""
        button_id = event.button.id
        if button_id == "submit_button":
            self.action_submit_text()
        elif button_id == "next_button" and self.thread:
            self.action_load_post()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Event handler called when an input is submitted."""
        input_id = event.input.id
        if input_id == "input_topic":
            topic = event.input.value.strip()
            self.generate_initial_post(topic)

    def action_toggle_comment(self) -> None:
        """Action to toggle the user-input area."""
        logging.info("[action_toggle_comment]")
        self._toggle("#userinput")

    def action_submit_text(self) -> None:
        """Action to submit text from the text area."""
        logging.info("[action_submit_text]")
        text_area = self.query_one(TextArea)
        comment = text_area.text.strip()
        if comment:
            if self.thread:
                post = chat.add_message(self.thread, comment)
                self._add_to_log(post)
            else:
                # Initialize the thread manually
                self.initialize_thread_manually(comment)
                if self.thread is None:
                    raise ValueError("Thread is None. Failed to initialize.")
            text_area.text = ""

    @work
    async def action_load_post(self) -> None:
        """Action to load the next post."""
        logging.info("[action_load_post]")
        if self.thread:
            # Show the loading indicator
            indicator = self.query_one(LoadingIndicator)
            indicator.display = True

            # Update the thread by generating a post
            post = await chat.update_thread_async(self.system, self.thread)

            # Update the display
            self._add_to_log(post)

            # Save the thread to a file
            p = Path(f"{PATH_BASE}_{self.thread.topic[:10]}_{self.saveid}.txt")
            chat.save_thread(p, self.thread)

            # Hide the loading indicator
            indicator.display = False

    def action_toggle_settings(self) -> None:
        """Action to toggle the settings."""
        logging.info("[action_toggle_settings]")
        self._toggle("#select_lang")

    @work
    async def generate_initial_post(self, topic: str) -> None:
        """Generate an initial post and a thread."""
        # Show the loading indicator
        logging.info("[generate_initial_post] %s", topic)
        indicator = self.query_one(LoadingIndicator)
        indicator.display = True

        # Generate the ininital post
        self.thread = await chat.init_thread_async(self.system, topic)

        # Initialize the thread display
        rich_log = self.query_one(RichLog)
        rich_log.clear()
        s = chat.format_thread(self.thread)
        rich_log.write(s)

        # Hide the topic-selection widget
        self._hide("#input_topic")
        self._hide("#vertical_topic")

        # Hide the loading indicator
        indicator.display = False

    def initialize_thread_manually(self, text: str) -> None:
        """Initialize thread manually."""
        # Generate the ininital post
        self.thread = chat.create_thread_from_text(text)

        # Initialize the thread display
        rich_log = self.query_one(RichLog)
        rich_log.clear()
        s = chat.format_thread(self.thread)
        rich_log.write(s)

        # Hide the topic-selection widget
        self._hide("#input_topic")
        self._hide("#vertical_topic")

    def show_input_topic(self) -> None:
        """Show the topic input box."""
        self._show("#input_topic")

    def _hide(self, selector) -> None:
        """Hide a widget."""
        widget = self.query_one(selector)
        widget.display = False

    def _show(self, selector) -> None:
        """Show a widget."""
        widget = self.query_one(selector)
        widget.display = True

    def _toggle(self, selector) -> None:
        """Toggle a widget."""
        widget = self.query_one(selector)
        widget.display = not widget.display

    def select_lang(self, lang: chat.Language) -> None:
        """Select a language."""
        self.lang = lang
        self.system.set_lang(lang)
        self._hide("#select_lang")
        logging.info("Language: %s", lang)

    def _add_to_log(self, post: chat.Post) -> None:
        """Add a string to the log."""
        rich_log = self.query_one(RichLog)
        s = chat.format_post_with_username(post)
        rich_log.write("\n\n")
        rich_log.write(s)
        logging.info("Comment: %s", post.text)

    @on(Select.Changed)
    def select_changed(self, event: Select.Changed) -> None:
        """Event handler called when the select widget changes."""
        logging.info("Called: select_changed")
        if event.select.id == "select_lang":
            self.select_lang(str(event.value).strip())  # type: ignore
        elif event.select.id == "select_topic":
            topic = str(event.value).strip()
            if topic == chat.OTHER_TOPIC:
                self.show_input_topic()
            else:
                self.generate_initial_post(str(event.value))


def main() -> None:
    app = Chan()
    app.run()


if __name__ == "__main__":
    main()
