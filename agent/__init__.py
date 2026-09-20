"""Turns a spelled message into real-world actions.

The speller produces a handful of words; an LLM turns them into a structured
tool call and Composio executes it against the user's own Google account.

    python -m agent.connect                     # one-time Google Calendar OAuth
    python -m agent.connect --toolkit gmail     # one-time Gmail OAuth
    python -m agent.runner "dentist 3pm"        # dry run
    python -m agent.runner "email sam i am late" # dry run
    python -m agent.runner "dentist 3pm" --live # writes a real event

Nothing here imports psychopy or brainflow, so it runs -- and is tested --
without a headset. ssvep_training/agent.py drives it from the speller UI.

Submodules are imported on use, not here: `python -m agent.runner` would
otherwise load runner twice and warn about it.
"""
from dotenv import load_dotenv

load_dotenv()          # OPENAI_API_KEY / COMPOSIO_API_KEY live in .env
