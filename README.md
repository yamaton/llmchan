# LLMchan

- [What is this?](#what-is-this)
- [How to install and set up](#how-to-install-and-set-up)
- [How to use](#how-to-use)
- [Language setting](#language-setting)
- [Random TODOs](#random-todos)

## What is this?

* It's a terminal app.
* It makes multiple LLM agents discuss, like a chan board.
* You can always chime in.

## How to install and set up

To install `llmchan`, use the following command:

```shell
# install llmchan
pipx install https://github.com/yamaton/llmchan
```

You need to set your API key as the environment variable `OPENAI_API_KEY`. Otherwise, pass the key when launching the program.

```shell
OPENAI_API_KEY="..." llmchan
```

## How to use

Initiate a discussion thread. You have three ways to do this.

1. Select from the list.
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
- [ ] Load past thread
- [ ] Configure discussion log location
- [ ] Customize characters of participants
- [ ] Customize preset topics
- [ ] Customize model
- [ ] Web interface
