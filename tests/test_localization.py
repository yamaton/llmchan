"""
Test localization module.
"""

import llmchan.localization as loc


def test_localization():
    """Check one-to-one correspondence of the data."""
    assert sorted(loc.LangPromptDict.keys()) == sorted(loc.LangList)
