"""
Record labelled SSVEP trials so TRCA can learn your specific responses.

All squares flash at once, one frequency each (frequencies are set in config.py).
    python -m ssvep_training.collect_training_data --blocks 8
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

from . import config as cfg
from .recording import RecordingProcess
from .session import encode_marker


class Quit(Exception):
    """Run ended early: Escape pressed, or the board never started streaming."""




def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", help="override PORT_PATH from .env")
    parser.add_argument("--blocks", type=int, default=8, help="times every square is cued (6-16 sensible)")
    parser.add_argument("--windowed", action="store_true", help="run in a window instead of fullscreen")
    parser.add_argument('--seed', type=int, default=20260920)
    parser.add_argument("--channels", default="",
                        help="board channels to record, e.g. 1,2,3,4 (default: all 8). Dead "
                             "electrodes also sit in the bias loop, so excluding them helps every channel.")
    parser.add_argument("--stimulus", choices=["flicker", "motion"], default=cfg.STIMULUS_MODE)
    args = parser.parse_args()
    cfg.configure_stimulus(args.stimulus)
    channels = [int(c) for c in args.channels.split(",")] if args.channels else None
    if args.blocks < 3:
        parser.error('use at least 3 blocks so each held-out fold retains repeated training trials')
    if channels is not None and (not channels or len(set(channels)) != len(channels)
                                 or not set(channels) <= set(range(1,9))):
        parser.error('channels must be unique integers from 1 through 8')
    rng = random.Random(args.seed)

    session_dir = os.path.join(cfg.TRAINING_DATA_DIR, time.strftime("session_%Y%m%d_%H%M%S"))
    os.makedirs(session_dir, exist_ok=True)
    trial_log = {'seed': args.seed, 'planned_blocks': args.blocks, 'trials': []}

    def save_log():
        temporary = os.path.join(session_dir, 'calibration.tmp.json')
        with open(temporary, 'w') as f:
            json.dump(trial_log, f, indent=2)
        os.replace(temporary, os.path.join(session_dir, 'calibration.json'))

    save_log()
    # Start the board first: its setup runs while the window opens.
    recorder = RecordingProcess(port=args.port, session_dir=session_dir, channels=channels)
    recorder.start()

    # PsychoPy is imported only in the display process, after the board process exists.
    from psychopy import core, visual
    from .stimulus import build_stimuli, build_window, run_trial, wait_for_board, wait_for_key, show_message

    win = build_window(fullscreen=not args.windowed)
    print("Press SPACE (or click) in the flicker window to continue at each screen; Escape quits.")
    completed = 0
    try:
        if not wait_for_key(win, cfg.stimulus_warning()):
            raise Quit
        if not wait_for_board(win, recorder, "Setting up the Knight board (~8 s)..."):
            raise Quit("the board never started streaming - see [board] above")
        flat = [row for row in range(1,9) if recorder.flat_eeg_mask.value & (1 << row)]
        if flat:
            raise Quit(f'channels {flat} are flat/nonfinite at startup; check channel enabling/contact before calibration')

        squares, cues, labels = build_stimuli(win)
        center = visual.TextStim(win, text='+', pos=(0, 0), color='gray', height=.06)
        trial_log['measured_refresh_hz'] = win.ssvep_refresh
        for block in range(1, args.blocks + 1):
            if not wait_for_key(win, f"Block {block} of {args.blocks}\n\n"
                                     "Look only at the square outlined in red.\n"
                                     "Blink during the breaks between trials.\n\n"
                                     "Press SPACE to start."):
                raise Quit
            # Random cue order per block, so the model can't learn order/fatigue
            # effects; each trial is labelled by target via its marker.
            order = list(range(cfg.N_TARGETS))
            rng.shuffle(order)
            for target in order:
                marker = encode_marker(block, target)
                show_message(win, f'Block {block}/{args.blocks}: look at {cfg.TARGET_LETTERS[target]}')
                core.wait(cfg.CUE_DURATION)
                dropped = win.nDroppedFrames
                done = run_trial(win, squares, cues, target, recorder, marker, overlay=[*labels, center])
                trial_log['trials'].append({'marker': marker, 'block': block, 'target': target,
                                           'completed': done, 'late_frames': win.nDroppedFrames-dropped})
                save_log()
                if not done:
                    raise Quit
            recorder.request_save()
            completed = block

        wait_for_key(win, f"Done! {completed} blocks recorded.\n\nPress SPACE to close.")
    except Quit as why:
        print(f"Stopped: {why or 'Escape pressed'}.")
    except KeyboardInterrupt:
        print("Stopped: Ctrl+C in the terminal.")
    finally:
        save_log()
        recorder.stop()
        recorder.join(timeout=15)   # final save happens as the board process exits
        win.close()
        summarize(session_dir, completed)
        core.quit()


def summarize(session_dir: str, completed: int) -> None:
    from .session import load_trials

    if not os.path.exists(os.path.join(session_dir, "raw.npz")):
        print("Nothing was recorded.")
        return
    trials = load_trials(session_dir, reject_bad=True)
    n = trials.eeg.shape[-1]
    print(f"\nSession saved: {session_dir}")
    print(f"  {completed} complete block(s), {n} usable trial(s)"
          + (f", {trials.skipped} rejected for timing/transport/data quality" if trials.skipped else ""))
    if n:
        print(f"Next: .venv/bin/python -m ssvep_training.evaluate_trca --stimulus {cfg.STIMULUS_MODE} --session {session_dir}")


if __name__ == "__main__":
    main()
