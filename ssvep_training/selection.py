"""Live TRCA decision policy, also used when estimating language evidence."""
import math


def trca_choice(choice, peak, minimum=.1, below='reject'):
    if below not in ('reject', 'b', 'winner'):
        raise ValueError('Unknown below-threshold policy')
    # Never convert missing/corrupt data into a fallback selection.
    if choice not in (0, 1) or peak is None or not math.isfinite(peak):
        return -1
    return choice if peak >= minimum or below == 'winner' else (1 if below == 'b' else -1)
