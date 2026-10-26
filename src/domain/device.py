"""A readable name for the browser behind a User-Agent, for the sessions list."""

import re

_BROWSERS = (
    ("Edge", re.compile(r"Edg(?:e|A|iOS)?/")),
    ("Opera", re.compile(r"OPR/|Opera")),
    ("Firefox", re.compile(r"Firefox/|FxiOS/")),
    ("Chrome", re.compile(r"Chrome/|CriOS/")),
    ("Safari", re.compile(r"Safari/")),
)
_SYSTEMS = (
    ("Android", re.compile(r"Android")),
    ("iOS", re.compile(r"iPhone|iPad|iPod")),
    ("Windows", re.compile(r"Windows")),
    ("macOS", re.compile(r"Macintosh|Mac OS X")),
    ("Linux", re.compile(r"Linux|X11")),
)


def describe_device(user_agent: str | None) -> dict[str, str]:
    """Browser and operating system, "" for what is not recognised.

    Order matters: Edge and Opera also say Chrome, Chrome also says Safari, and Android
    also says Linux. Anything unknown is left empty rather than guessed.
    """
    agent = user_agent or ""
    browser = next((name for name, rx in _BROWSERS if rx.search(agent)), "")
    system = next((name for name, rx in _SYSTEMS if rx.search(agent)), "")
    return {"browser": browser, "os": system}
