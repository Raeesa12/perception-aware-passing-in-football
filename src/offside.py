"""
Layer 2 helper: Offside Detection

Determines, for a given pass and its 360 freeze frame, which candidate
teammates were onside vs. offside at the moment of the pass.

Attacking direction is inferred per-pass from the opposing goalkeeper's
position in the same freeze frame (via the 'keeper' flag), rather than
assumed globally per team/period -- this is robust to StatsBomb's raw,
un-normalized pitch coordinates.

The offside line is the position of the second-deepest defender (the
goalkeeper is included in that ordering, not excluded by default -- in
practice they are usually deepest, but are not assumed to be).
"""

import numpy as np
import pandas as pd


def get_opponent_keeper(frame_at_event):
    """
    Return the opposing goalkeeper's row from a 360 freeze frame, if visible.
    Returns None if not present in this frame.
    """
    keeper_rows = frame_at_event[
        (frame_at_event['teammate'] == False) & (frame_at_event['keeper'] == True)
    ]
    if keeper_rows.empty:
        return None
    return keeper_rows.iloc[0]


def compute_offside_line(frame_at_event, attacking_toward_x):
    """
    Given all players in a 360 frame and the x-coordinate the attacking team
    is attacking toward, return the offside line x-coordinate: the position
    of the second-deepest defender (goalkeeper included in the sort).

    Returns None if fewer than 2 defenders are visible in the frame.
    """
    defenders = frame_at_event[frame_at_event['teammate'] == False].copy()
    if len(defenders) < 2:
        return None

    defenders['x'] = defenders['location'].apply(lambda loc: loc[0])

    if attacking_toward_x > 60:
        sorted_defenders = defenders.sort_values('x', ascending=False)
    else:
        sorted_defenders = defenders.sort_values('x', ascending=True)

    return sorted_defenders.iloc[1]['x']


def is_offside(candidate_x, ball_x, offside_line_x, attacking_toward_x):
    """
    A candidate is offside if they are beyond BOTH the ball and the
    second-deepest defender, on the attacking side.

    If offside_line_x is None (couldn't be determined), defaults to
    "not offside" -- a conservative choice that avoids incorrectly
    excluding a candidate when the defensive line couldn't be established.
    """
    if offside_line_x is None:
        return False

    if attacking_toward_x > 60:
        return candidate_x > ball_x and candidate_x > offside_line_x
    else:
        return candidate_x < ball_x and candidate_x < offside_line_x


def classify_offside(frame_at_event, ball_loc):
    """
    Full pipeline for one pass: given its 360 freeze frame and ball location,
    determine attacking direction from the opponent keeper, compute the
    offside line, and classify every teammate in the frame as onside/offside.

    Returns a DataFrame of teammate rows with two added columns:
    'attacking_toward_x' and 'offside'. Returns None if the opponent
    keeper isn't visible in this frame (attacking direction can't be
    determined).
    """
    keeper_row = get_opponent_keeper(frame_at_event)
    if keeper_row is None:
        return None

    attacking_toward_x = keeper_row['location'][0]
    offside_line_x = compute_offside_line(frame_at_event, attacking_toward_x)

    teammates = frame_at_event[frame_at_event['teammate'] == True].copy()
    if teammates.empty:
        return teammates

    teammates['attacking_toward_x'] = attacking_toward_x
    teammates['offside'] = teammates['location'].apply(
        lambda loc: is_offside(loc[0], ball_loc[0], offside_line_x, attacking_toward_x)
    )
    return teammates