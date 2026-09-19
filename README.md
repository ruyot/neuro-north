# EEG Thought-to-Text

## The idea

This hackathon project explores thought-to-text using an EEG headset, which measures small electrical signals from the brain. The goal is to help someone produce text without speaking or typing.

## What the code does today

The tools connect to a NeuroPawn Knight headset and show live brain and movement readings on a computer. Both tools can also use made-up data, so they can be tried without a headset.

## Files

- [`stream_test.py`](stream_test.py) — Prints live readings as numbers to check the headset connection.
- [`live_plot.py`](live_plot.py) — Shows scrolling graphs of the readings, with an option to reduce unwanted noise in the brain signals.
- [`requirements.txt`](requirements.txt) — Lists the Python software needed to run the tools.
- [`.gitignore`](.gitignore) — Keeps local setup files and generated data out of the shared repo.

## Current stage

This is the headset connection and viewing part of the project. It does **not yet save recordings, recognise words, or turn thoughts into text**. Thought-to-text is the goal, not a working feature yet.
