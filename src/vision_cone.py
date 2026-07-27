"""
Layer 1: Geometric Vision-Cone Filter

Approximates a player's feasible visual field at the moment of a pass using
a 120-degree cone (Arbues-Sanguesa et al. 2020), and classifies each teammate
in the StatsBomb 360 freeze frame as visible or invisible relative to that cone.

Note: StatsBomb open data does not include a body-orientation attribute.
As a substitute, orientation is approximated from the player's movement vector
into the pass (previous touch -> pass location). This is a documented
methodological adaptation from the original proposal.
"""

import numpy as np
import pandas as pd


def compute_orientation_vector(row):
    """
    Approximate a player's facing direction using the vector from their
    previous touch to their current pass location.

    Expects `row` to have 'prev_location' and 'location' columns, each a
    [x, y] list or None.

    Returns a unit vector (dx, dy), or None if it cannot be computed.
    """
    if row['prev_location'] is None or not isinstance(row['prev_location'], list):
        return None
    if row['location'] is None or not isinstance(row['location'], list):
        return None

    dx = row['location'][0] - row['prev_location'][0]
    dy = row['location'][1] - row['prev_location'][1]
    norm = np.sqrt(dx**2 + dy**2)
    if norm == 0:
        return None

    return (dx / norm, dy / norm)


def angle_to_teammate(passer_loc, passer_orientation, teammate_loc):
    """
    Compute the angle (in degrees) between a passer's orientation vector
    and the vector from the passer to a given teammate.

    0 degrees = directly ahead, 180 degrees = directly behind.
    """
    v_passer = np.array(passer_orientation)
    v_teammate = np.array([
        teammate_loc[0] - passer_loc[0],
        teammate_loc[1] - passer_loc[1]
    ])

    norm_teammate = np.linalg.norm(v_teammate)
    if norm_teammate == 0:
        return None

    v_teammate_unit = v_teammate / norm_teammate
    cos_angle = np.clip(np.dot(v_passer, v_teammate_unit), -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def get_teammates_in_frame(match_id, event_id, frames_df):
    """
    Return all non-actor teammate rows from the 360 freeze frame for a
    given event.
    """
    frame_rows = frames_df[frames_df['id'] == event_id]
    return frame_rows[(frame_rows['teammate'] == True) & (frame_rows['actor'] == False)]


def classify_pass_visibility(passer_loc, passer_orientation, match_id, event_id,
                              frames_df, cone_half_angle=60):
    """
    For one pass, return all teammates in its 360 frame with their angle
    to the passer's orientation vector and a binary visibility flag
    (True if within the cone_half_angle, i.e. inside the full 2*cone_half_angle cone).

    Returns None if there is no 360 frame data for this event.
    """
    teammates = get_teammates_in_frame(match_id, event_id, frames_df)

    if teammates.empty:
        return None

    teammates = teammates.copy()
    teammates['angle_to_orientation'] = teammates['location'].apply(
        lambda loc: angle_to_teammate(passer_loc, passer_orientation, loc)
    )
    teammates = teammates.dropna(subset=['angle_to_orientation'])
    teammates['visible'] = teammates['angle_to_orientation'] <= cone_half_angle

    return teammates


def add_orientation_vectors(events_df):
    """
    Given a raw StatsBomb events dataframe, add 'prev_location' and
    'orientation_vector' columns, ordered by possession and event index.

    Returns the full sorted events dataframe (not just passes) so it can be
    reused for other event types later if needed.
    """
    events_sorted = events_df.sort_values(['possession', 'index']).reset_index(drop=True)
    events_sorted['prev_location'] = events_sorted.groupby('player_id')['location'].shift(1)
    events_sorted['orientation_vector'] = events_sorted.apply(compute_orientation_vector, axis=1)
    return events_sorted


def build_pass_visibility_table(passes_df, frames_df):
    """
    Run classify_pass_visibility over every pass with a valid orientation
    vector, and return a summary dataframe with per-pass counts:
    n_teammates, n_visible, n_invisible, has_360_frame.
    """
    results = []
    valid_passes = passes_df[passes_df['orientation_vector'].notna()]

    for _, row in valid_passes.iterrows():
        classified = classify_pass_visibility(
            passer_loc=row['location'],
            passer_orientation=row['orientation_vector'],
            match_id=row['match_id'],
            event_id=row['id'],
            frames_df=frames_df
        )

        if classified is None or classified.empty:
            results.append({
                'pass_id': row['id'],
                'has_360_frame': False,
                'n_teammates': 0,
                'n_visible': 0,
                'n_invisible': 0
            })
        else:
            results.append({
                'pass_id': row['id'],
                'has_360_frame': True,
                'n_teammates': len(classified),
                'n_visible': int(classified['visible'].sum()),
                'n_invisible': int((~classified['visible']).sum())
            })

    return pd.DataFrame(results)
