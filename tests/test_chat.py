"""
Test chat module
"""

import llmchan.chat as chat


def test___parse_thread_header():
    """Test _parse_thread_header function."""
    f = chat._parse_thread_header

    assert f("[0] OP -> [1,2]") == (0, "OP", [1, 2])
    assert f("[13] baba_keke -> [11]") == (13, "baba_keke", [11])
    assert f("[3] OP ") == (3, "OP", [])


def test___parse_single_post():
    """Test _parse_single_post function."""
    f = chat._parse_single_post
    post_str = "[0] OP\nHello, world!"
    post = chat.Post(id=0, username="OP", text="Hello, world!")
    assert f(post_str) == post

    post_str = "[3] Foo_Bar -> [1]\nbaba\nkeke\n"
    post = chat.Post(id=3, username="Foo_Bar", text="baba\nkeke", in_reply_to=[1])
    assert f(post_str) == post

    post_str = "[223] Mr. Keke -> [222]\n..."
    post = chat.Post(id=223, username="Mr. Keke", text="...", in_reply_to=[222])
    assert f(post_str) == post


def test__create_thread_from_text():
    """Test create_thread_from_text function."""
    text = "alpha bravo\ncharlie delta\n"
    thread = chat.create_thread_from_text(text)
    assert len(thread.posts) == 1
    assert thread.posts[0].text == text.strip()
    assert thread.posts[0].id == 0
    assert thread.posts[0].username == "OP"

