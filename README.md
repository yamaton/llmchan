# LLMchan

![](images/screenshot0.png)

- [What is this?](#what-is-this)
- [How to install and set up](#how-to-install-and-set-up)
- [How to use](#how-to-use)
- [Language setting](#language-setting)
- [Random TODOs](#random-todos)

## What is this?

- It's a terminal app using OpenAI's Chat API.
- It makes multiple LLM agents discuss, like a chan board.
- You can always chime in the discussion.

## How to install and set up

To install `llmchan`, use the following command. It requires [`pipx`](https://github.com/pypa/pipx).

```shell
pipx install https://github.com/yamaton/llmchan
```

Please make [your own OPENAI API key](https://platform.openai.com/api-keys) available as the environment variable `OPENAI_API_KEY` before running the program.

```shell
# Make $OPENAI_API_KEY available
llmchan
```

I personally use the CLI [`dotenv`](https://pypi.org/project/python-dotenv/) to load the environment variable dynamically from a local file `.env`.

```shell
# Run llmchan after loading environment variables from .env file
dotenv run llmchan
```

## How to use

Initiate a discussion thread. You have three ways to do this.

1. Select a topic from the list.
2. Enter a topic.
3. Enter full text as a thread-opening post.

If you go with #1 or #2, the initial post is generated based on your selection or topic.

Once a thread is started, you can:

1. Generate the next post.
2. Enter your own post.

**NOTE:** Threads are automatically saved with the prefix `thread_` in the current directory.

## Language setting

You can set the language used in the discussion. Type `i` to open the language selection menu.

For a list of supported languages, refer to the [Steam Localization Languages](https://partner.steamgames.com/doc/store/localization/languages). [Note: Still in progress.]

## Random TODOs

- [ ] Save logs in `~/.local/share/llmchan`
- [ ] Load a thread
- [ ] Respect `$LANGUAGE` by default
- [ ] Remember language setting
- [ ] Customize saved log location
- [ ] Customize participant characteristics/prompts
- [ ] Customize preset topics
- [ ] Customize models
- [ ] Web interface
