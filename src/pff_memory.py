"""
RQ3 with PFF tracking: what could the passer have remembered?

For every PFF pass, the passer's vision cone (120 degrees, centred on their
movement direction, as in Layer 1) is rebuilt at every 10 Hz sample of the
seconds before the pass. A teammate counts as SEEN at a sample if they were
inside the passer's cone at that moment. For each teammate outside the cone
at the moment of the pass, the memory history is the list of samples in the
last H seconds at which the passer saw them: the only information a memory
could hold.

All positions are in metres, rotated so the passing team attacks toward +x
(PFF origin at the centre spot).
"""
import numpy as np
import pandas as pd

import pff

HALF_ANGLE = 60.0
H_SECONDS = 5.0
K_MAX = 20          # most recent seen samples kept (2 s of continuous viewing at 10 Hz)
STEP = 3            # tracking frames per sample (10 Hz)


def _angles(origin, heading, pts):
    """Angle in degrees between heading (2,) and the vectors origin -> pts (n, 2)."""
    v = pts - origin
    n = np.hypot(v[:, 0], v[:, 1])
    cos = (v @ heading) / np.maximum(n, 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def build_examples(arr, passes_g, fps, h_seconds=H_SECONDS, k_max=K_MAX, half_angle=HALF_ANGLE):
    """
    arr: output of pff.extract_match_arrays for one game.
    passes_g: rows of pff_passes_all.pkl for the game (passes and crosses).

    Returns (mates, hist):
      mates: one row per teammate (passer excluded) at every pass, with
             in_cone, target position and visibility at the pass, receiver
             flags, number of seen samples, time since last seen, last seen
             position and last-seen velocity estimate.
      hist:  (N, k_max, 3) float32 seen history (x, y, seconds before the pass),
             right-aligned, NaN padded; row i belongs to mates row i.
    """
    frame = arr['frame']
    xy = arr['xy']
    head, _ = pff.headings(xy, frame=frame, step=STEP)
    pid_col = {int(p): j for j, p in enumerate(arr['player_id'])}
    W = int(round(h_seconds * fps / STEP))
    rows, hists = [], []
    empty_hist = np.full((k_max, 3), np.nan, dtype=np.float32)

    for r in passes_g.itertuples():
        if r.attacking_direction not in ('R', 'L') or pd.isna(r.passer_id):
            continue
        f0 = int(round(r.event_time * fps))
        t0 = int(np.searchsorted(frame, f0))
        cands = [t for t in (t0 - 1, t0) if 0 <= t < len(frame)]
        if not cands:
            continue
        t0 = min(cands, key=lambda t: abs(frame[t] - f0))
        if abs(frame[t0] - f0) > STEP:
            continue
        jp = pid_col.get(int(r.passer_id))
        if jp is None or np.isnan(xy[t0, jp, 0]):
            continue
        sign = 1.0 if r.attacking_direction == 'R' else -1.0
        same = arr['home'] == arr['home'][jp]
        mates = [j for j in np.flatnonzero(same) if j != jp and not np.isnan(xy[t0, j, 0])]
        if not mates:
            continue
        h0 = head[t0, jp]
        heading_fallback = bool(np.isnan(h0[0]))
        if heading_fallback:
            h0 = np.array([sign, 0.0])          # facing the opponent goal
        p0 = xy[t0, jp]
        ang0 = _angles(p0, h0, xy[t0, mates])

        # the window before the pass, contiguous in time
        lo = max(0, t0 - W)
        ts = np.arange(lo, t0)
        ts = ts[(f0 - frame[ts]) <= h_seconds * fps]
        for m_i, j in enumerate(mates):
            in_cone = bool(ang0[m_i] <= half_angle)
            row = {
                'pff_game_id': r.pff_game_id, 'possession_event_id': r.possession_event_id,
                'passer_id': int(r.passer_id), 'mate_id': int(arr['player_id'][j]),
                'is_receiver': (not pd.isna(r.receiver_id)) and int(r.receiver_id) == int(arr['player_id'][j]),
                'is_target': (not pd.isna(r.target_id)) and int(r.target_id) == int(arr['player_id'][j]),
                'complete': r.outcome == 'C', 'event_type': r.event_type,
                'in_cone': in_cone, 'angle': float(ang0[m_i]), 'heading_fallback': heading_fallback,
                'x': sign * float(xy[t0, j, 0]), 'y': sign * float(xy[t0, j, 1]),
                'passer_x': sign * float(p0[0]), 'passer_y': sign * float(p0[1]),
                'target_visible': bool(arr['vis'][t0, j] == 1),
                'passer_visible': bool(arr['vis'][t0, jp] == 1),
            }
            hist = empty_hist
            if not in_cone and len(ts):
                pp = xy[ts, jp]
                hh = head[ts, jp]
                pm = xy[ts, j]
                ok = ~np.isnan(pp[:, 0]) & ~np.isnan(hh[:, 0]) & ~np.isnan(pm[:, 0])
                seen = np.zeros(len(ts), dtype=bool)
                if ok.any():
                    v = pm[ok] - pp[ok]
                    n = np.hypot(v[:, 0], v[:, 1])
                    cos = (v * hh[ok]).sum(1) / np.maximum(n, 1e-9)
                    seen[ok] = np.degrees(np.arccos(np.clip(cos, -1, 1))) <= half_angle
                st = ts[seen]
                row['n_seen'] = int(len(st))
                if len(st):
                    keep = st[-k_max:]
                    hist = np.full((k_max, 3), np.nan, dtype=np.float32)
                    hist[k_max - len(keep):, 0] = sign * xy[keep, j, 0]
                    hist[k_max - len(keep):, 1] = sign * xy[keep, j, 1]
                    hist[k_max - len(keep):, 2] = (f0 - frame[keep]) / fps
                    row['gap_s'] = float((f0 - frame[st[-1]]) / fps)
                    row['last_x'], row['last_y'] = float(hist[-1, 0]), float(hist[-1, 1])
                    row['last_visible'] = bool(arr['vis'][st[-1], j] == 1)
            rows.append(row)
            hists.append(hist)
    mates_df = pd.DataFrame(rows)
    return mates_df, (np.stack(hists) if hists else np.empty((0, k_max, 3), dtype=np.float32))


def back_line_gaps(arr, home_start_left, gk_ids, every=10):
    """
    Lateral gaps (metres) between neighbouring players of a team's back line,
    sampled every `every` samples (1 s at 10 Hz) in periods 1 and 2, when the
    ball is in that team's own half. The back line is the four deepest
    outfield players; a sample is used only when all four are VISIBLE.
    Used to set the memory threshold tau from the data, as the proposal
    specifies (defensive line widths).
    """
    is_gk = np.array([int(p) in gk_ids for p in arr['player_id']])
    out = []
    for t in range(0, len(arr['frame']), every):
        per = int(arr['period'][t])
        bx = arr['ball'][t, 0]
        if per not in (1, 2) or np.isnan(bx):
            continue
        for home in (True, False):
            starts_left = home_start_left if home else not home_start_left
            defends_left = starts_left if per == 1 else not starts_left
            if (bx < 0) != defends_left:
                continue
            cols = np.flatnonzero((arr['home'] == home) & ~is_gk & (arr['vis'][t] >= 0))
            if len(cols) < 9:
                continue
            x, y, v = arr['xy'][t, cols, 0], arr['xy'][t, cols, 1], arr['vis'][t, cols]
            depth = x if defends_left else -x
            back = np.argsort(depth)[:4]
            if (v[back] != 1).any():
                continue
            out.extend(np.diff(np.sort(y[back])))
    return np.array(out)


def state_at_pass(arr, f0, sign, lag=2, fps=29.97, max_speed=12.0):
    """
    Positions and velocities of every player at the sample nearest frame f0,
    rotated so the passing team attacks toward +x (sign = +1 for 'R', -1 for
    'L'). Velocity = (position now - position `lag` samples earlier) / time,
    capped at max_speed; zero if the earlier sample is missing or not
    contiguous. Returns (t0, pos (P, 2), vel (P, 2), present (P,)) or None.
    """
    frame = arr['frame']
    t0 = int(np.searchsorted(frame, f0))
    cands = [t for t in (t0 - 1, t0) if 0 <= t < len(frame)]
    if not cands:
        return None
    t0 = min(cands, key=lambda t: abs(frame[t] - f0))
    if abs(frame[t0] - f0) > STEP:
        return None
    pos = sign * arr['xy'][t0].astype(float)
    present = ~np.isnan(pos[:, 0])
    vel = np.zeros_like(pos)
    if t0 - lag >= 0 and frame[t0] - frame[t0 - lag] == lag * STEP:
        prev = sign * arr['xy'][t0 - lag].astype(float)
        ok = present & ~np.isnan(prev[:, 0])
        vel[ok] = (pos[ok] - prev[ok]) / (lag * STEP / fps)
    sp = np.hypot(vel[:, 0], vel[:, 1])
    vel *= np.minimum(1.0, max_speed / np.maximum(sp, 1e-9))[:, None]
    return t0, pos, vel, present
