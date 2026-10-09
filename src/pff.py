"""
PFF FC World Cup 2022 helpers: loading, coordinates and linking to StatsBomb.

The PFF data sits in data/pff/ as extracted from Google Drive (folder names
carry a download timestamp, so files are found with glob):
    data/pff/Metadata*/Metadata/{game_id}.json
    data/pff/Rosters*/Rosters/{game_id}.json
    data/pff/Event Data*/Event Data/{game_id}.json          (latest version)
    data/pff/Tracking Data*/Tracking Data/{game_id}.jsonl.bz2

PFF coordinates: metres, origin at the centre spot, x along the touchlines
(+-52.5), y across (+-34), as seen by the main camera. Each event carries
teamAttackingDirection ('R' or 'L') for the team in possession.

StatsBomb coordinates (what the rest of the project uses): yards, 120 x 80,
origin top-left, the acting team attacking toward x = 120, y increasing
downward. pff_to_sb converts one into the other, for any player or ball
position, given the direction of the team in possession.

Every off-ball PFF player position has a visibility flag. 'ESTIMATED'
positions are guessed by the tracking provider when the player is off
screen, so they are not ground truth.
"""

import bz2
import glob
import json
import os
import unicodedata

import numpy as np
import pandas as pd

PFF_DIR = os.path.join('..', 'data', 'pff')
SB_LENGTH, SB_WIDTH = 120.0, 80.0


# ---------------------------------------------------------------- file paths
def _find(pattern):
    hits = glob.glob(os.path.join(PFF_DIR, pattern))
    if not hits:
        raise FileNotFoundError(os.path.join(PFF_DIR, pattern))
    return hits[0]


def metadata_path(gid):
    return _find(os.path.join('Metadata*', 'Metadata', f'{gid}.json'))


def roster_path(gid):
    return _find(os.path.join('Rosters*', 'Rosters', f'{gid}.json'))


def events_path(gid):
    return _find(os.path.join('Event Data*', 'Event Data', f'{gid}.json'))


def tracking_path(gid):
    return _find(os.path.join('Tracking Data*', 'Tracking Data', f'{gid}.jsonl.bz2'))


# ---------------------------------------------------------------- loading
def load_metadata():
    """One row per PFF game: id, teams, date, fps, pitch size."""
    rows = []
    for f in glob.glob(os.path.join(PFF_DIR, 'Metadata*', 'Metadata', '*.json')):
        m = json.load(open(f, encoding='utf-8'))[0]
        pitch = m['stadium']['pitches'][0]
        rows.append({
            'pff_game_id': int(m['id']),
            'home_team': m['homeTeam']['name'],
            'away_team': m['awayTeam']['name'],
            'date': m['date'],
            'fps': m.get('fps', 29.97),
            'home_start_left': m.get('homeTeamStartLeft'),
            'pitch_length': pitch['length'],
            'pitch_width': pitch['width'],
        })
    return pd.DataFrame(rows).sort_values('pff_game_id').reset_index(drop=True)


def load_roster(gid):
    r = json.load(open(roster_path(gid), encoding='utf-8'))
    return pd.DataFrame([{
        'pff_player_id': int(x['player']['id']),
        'player_name': x['player']['nickname'],
        'team': x['team']['name'],
        'jersey': int(x['shirtNumber']),
        'position_group': x.get('positionGroupType'),
        'started': x.get('started'),
    } for x in r])


def load_events(gid):
    """The raw PFF event list for one game."""
    return json.load(open(events_path(gid), encoding='utf-8'))


def pff_passes(gid, events=None):
    """
    One row per PFF pass or cross, with the fields needed for linking and
    for the later RQ2/RQ3 checks. Times are video seconds.
    """
    if events is None:
        events = load_events(gid)
    rows = []
    for e in events:
        pe = e.get('possessionEvents') or {}
        t = pe.get('possessionEventType')
        if t not in ('PA', 'CR'):
            continue
        ge = e.get('gameEvents') or {}
        sm = e.get('stadiumMetadata') or {}
        ball = (e.get('ball') or [{}])[0]
        is_cross = t == 'CR'
        rows.append({
            'pff_game_id': gid,
            'game_event_id': e.get('gameEventId'),
            'possession_event_id': e.get('possessionEventId'),
            'event_type': t,
            'period': ge.get('period'),
            'event_time': e.get('eventTime'),
            'team': ge.get('teamName'),
            'home_team': ge.get('homeTeam'),
            'passer_id': pe.get('crosserPlayerId') if is_cross else pe.get('passerPlayerId'),
            'passer_name': pe.get('crosserPlayerName') if is_cross else pe.get('passerPlayerName'),
            'target_id': pe.get('targetPlayerId'),
            'receiver_id': pe.get('receiverPlayerId'),
            'outcome': pe.get('crossOutcomeType') if is_cross else pe.get('passOutcomeType'),
            'pass_type': pe.get('passType'),
            'pressure_type': pe.get('pressureType'),
            'better_option_type': pe.get('betterOptionType'),
            'better_option_player_id': pe.get('betterOptionPlayerId'),
            'better_option_time': pe.get('betterOptionTime'),
            'attacking_direction': sm.get('teamAttackingDirection'),
            'pitch_length': sm.get('pitchLength', 105.0),
            'pitch_width': sm.get('pitchWidth', 68.0),
            'ball_x': ball.get('x'),
            'ball_y': ball.get('y'),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- coordinates
def pff_to_sb(x, y, direction, length=105.0, width=68.0):
    """
    PFF metres (centre origin, camera view) -> StatsBomb yards for the team
    in possession attacking toward x = 120.

    direction 'R': the team attacks toward +x in PFF, so only a shift, a
    rescale and a y flip (PFF y points up, StatsBomb y points down).
    direction 'L': the same after a 180 degree rotation.
    Works on scalars or numpy arrays (direction may be an array too).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    right = np.asarray(direction) == 'R'
    sx = SB_LENGTH / np.asarray(length, dtype=float)
    sy = SB_WIDTH / np.asarray(width, dtype=float)
    half_l = np.asarray(length, dtype=float) / 2
    half_w = np.asarray(width, dtype=float) / 2
    sb_x = np.where(right, (x + half_l) * sx, (half_l - x) * sx)
    sb_y = np.where(right, (half_w - y) * sy, (y + half_w) * sy)
    return sb_x, sb_y


# ---------------------------------------------------------------- tracking
def read_tracking_frames(gid, frame_nums=None, smoothed=False):
    """
    Stream a tracking file and return {frameNum: frame dict}. If frame_nums
    is given, only those frames are kept (the file is still read in full). smoothed=True swaps in the Kalman
    smoothed player positions.
    """
    keep = None if frame_nums is None else set(int(f) for f in frame_nums)
    out = {}
    with bz2.open(tracking_path(gid), 'rt') as f:
        for line in f:
            d = json.loads(line)
            fn = d.get('frameNum')
            if keep is not None and fn not in keep:
                continue
            if smoothed:
                d['homePlayers'] = d.get('homePlayersSmoothed') or d.get('homePlayers')
                d['awayPlayers'] = d.get('awayPlayersSmoothed') or d.get('awayPlayers')
            for k in ('homePlayersSmoothed', 'awayPlayersSmoothed', 'ballsSmoothed'):
                d.pop(k, None)
            out[fn] = d
            if keep is not None and len(out) == len(keep):
                break
    return out


def time_to_frame(video_seconds, fps=29.97):
    return int(round(video_seconds * fps))


# ---------------------------------------------------------------- linking
def _ts_seconds(ts):
    h, m, s = ts.split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)


def strip_accents(s):
    if not isinstance(s, str):
        return s
    return ''.join(c for c in unicodedata.normalize('NFKD', s)
                   if not unicodedata.combining(c)).lower()


def _period_offset(sb_t, sb_team, pff_t, pff_team, bin_width=0.5):
    """
    Seconds to add to PFF video time to get StatsBomb period time. Taken as
    the most common difference between same-team passes (a histogram mode),
    so no clock metadata is needed and extra time works the same way.
    """
    d = sb_t[:, None] - pff_t[None, :]
    same = sb_team[:, None] == pff_team[None, :]
    d = d[same]
    if d.size == 0:
        return np.nan
    lo, hi = np.floor(d.min()), np.ceil(d.max()) + bin_width
    counts, edges = np.histogram(d, bins=np.arange(lo, hi + bin_width, bin_width))
    k = counts.argmax()
    centre = d[(d >= edges[k] - bin_width) & (d < edges[k + 1] + bin_width)]
    return float(np.median(centre))


def _greedy_one_to_one(cost):
    """Match rows to columns by increasing cost, each used at most once."""
    pairs = []
    if cost.size == 0:
        return pairs
    flat = np.argsort(cost, axis=None)
    used_r, used_c = set(), set()
    for idx in flat:
        r, c = divmod(int(idx), cost.shape[1])
        if not np.isfinite(cost[r, c]):
            break
        if r in used_r or c in used_c:
            continue
        pairs.append((r, c))
        used_r.add(r)
        used_c.add(c)
    return pairs


def link_passes(sb_events, pff_pass_df, max_dt=3.0, max_dist=20.0):
    """
    Link StatsBomb passes to PFF passes/crosses for one match.

    1. Per period, find the clock offset between the two providers.
    2. First pass: match on team, time and start location only.
    3. Build a StatsBomb player_id -> PFF player id map by majority vote.
    4. Final pass: also require the passer to agree wherever the map knows
       the player, then match one-to-one by cost = |dt| + distance / 10.

    Returns (links DataFrame, player map DataFrame).
    """
    s = sb_events[sb_events['type'] == 'Pass'].copy()
    s['t'] = s['timestamp'].map(_ts_seconds)
    s['loc_x'] = s['location'].str[0]
    s['loc_y'] = s['location'].str[1]
    p = pff_pass_df.dropna(subset=['ball_x', 'ball_y', 'attacking_direction']).copy()
    p = p[p['attacking_direction'].isin(['R', 'L'])]
    p['sb_x'], p['sb_y'] = pff_to_sb(p['ball_x'], p['ball_y'], p['attacking_direction'],
                                     p['pitch_length'], p['pitch_width'])

    def match(require_passer, pmap):
        out = []
        for per in sorted(set(s['period']) & set(p['period'])):
            a = s[s['period'] == per]
            b = p[p['period'] == per]
            off = _period_offset(a['t'].values, a['team'].values,
                                 b['event_time'].values, b['team'].values)
            dt = a['t'].values[:, None] - (b['event_time'].values[None, :] + off)
            dd = np.hypot(a['loc_x'].values[:, None] - b['sb_x'].values[None, :],
                          a['loc_y'].values[:, None] - b['sb_y'].values[None, :])
            ok = (a['team'].values[:, None] == b['team'].values[None, :]) \
                & (np.abs(dt) <= max_dt) & (dd <= max_dist)
            if require_passer:
                mapped = a['player_id'].map(pmap).values
                agree = (mapped[:, None] == b['passer_id'].values[None, :]) | pd.isna(mapped)[:, None]
                ok &= agree
            cost = np.where(ok, np.abs(dt) + dd / 10.0, np.inf)
            for r, c in _greedy_one_to_one(cost):
                out.append({
                    'sb_event_id': a['id'].values[r],
                    'pff_game_event_id': b['game_event_id'].values[c],
                    'pff_possession_event_id': b['possession_event_id'].values[c],
                    'period': per,
                    'clock_offset': off,
                    'dt': dt[r, c],
                    'dist_yd': dd[r, c],
                    'sb_player_id': a['player_id'].values[r],
                    'sb_player': a['player'].values[r],
                    'pff_passer_id': b['passer_id'].values[c],
                    'pff_passer': b['passer_name'].values[c],
                })
        return pd.DataFrame(out)

    first = match(False, {})
    votes = first.groupby(['sb_player_id', 'pff_passer_id']).size().rename('n').reset_index()
    votes = votes.sort_values('n', ascending=False)
    tot = votes.groupby('sb_player_id')['n'].transform('sum')
    votes['share'] = votes['n'] / tot
    pmap_df = votes.drop_duplicates('sb_player_id')
    pmap_df = pmap_df[(pmap_df['n'] >= 3) & (pmap_df['share'] >= 0.6)]
    pmap = dict(zip(pmap_df['sb_player_id'], pmap_df['pff_passer_id']))
    final = match(True, pmap)
    return final, pmap_df.rename(columns={'pff_passer_id': 'pff_player_id'})


# ---------------------------------------------------------------- player snapshots at events
def event_players(gid, events=None, possession_event_ids=None):
    """
    One row per player in PFF's snapshot at each pass or cross (the
    homePlayers / awayPlayers lists stored with every event), converted to
    StatsBomb yards for the team in possession attacking toward x = 120.

    Columns: possession_event_id, passing_team (bool), pff_player_id, jersey,
    position_group, visibility, confidence, speed (m/s), x, y (PFF metres),
    sb_x, sb_y (StatsBomb yards).
    """
    if events is None:
        events = load_events(gid)
    keep = None if possession_event_ids is None else set(possession_event_ids)
    rows = []
    for e in events:
        pe = e.get('possessionEvents') or {}
        if pe.get('possessionEventType') not in ('PA', 'CR'):
            continue
        pid = e.get('possessionEventId')
        if keep is not None and pid not in keep:
            continue
        ge = e.get('gameEvents') or {}
        sm = e.get('stadiumMetadata') or {}
        direction = sm.get('teamAttackingDirection')
        if direction not in ('R', 'L'):
            continue
        home_ball = ge.get('homeTeam')
        for side, players in (('home', e.get('homePlayers')), ('away', e.get('awayPlayers'))):
            for p in players or []:
                rows.append({
                    'possession_event_id': pid,
                    'passing_team': (side == 'home') == bool(home_ball),
                    'pff_player_id': p.get('playerId'),
                    'jersey': p.get('jerseyNum'),
                    'position_group': p.get('positionGroupType'),
                    'visibility': p.get('visibility'),
                    'confidence': p.get('confidence'),
                    'speed': p.get('speed'),
                    'x': p.get('x'),
                    'y': p.get('y'),
                    'direction': direction,
                    'pitch_length': sm.get('pitchLength', 105.0),
                    'pitch_width': sm.get('pitchWidth', 68.0),
                })
    df = pd.DataFrame(rows)
    if len(df):
        df['sb_x'], df['sb_y'] = pff_to_sb(df['x'], df['y'], df['direction'],
                                           df['pitch_length'], df['pitch_width'])
    return df


def match_points(a_xy, b_xy, max_dist=np.inf):
    """
    One-to-one nearest matching of points a to points b (minimum total
    distance). Uses scipy's Hungarian algorithm when available, otherwise a
    greedy match by increasing distance. Returns a list of (i, j, distance)
    with distance <= max_dist.
    """
    a_xy = np.asarray(a_xy, dtype=float).reshape(-1, 2)
    b_xy = np.asarray(b_xy, dtype=float).reshape(-1, 2)
    if len(a_xy) == 0 or len(b_xy) == 0:
        return []
    d = np.hypot(a_xy[:, None, 0] - b_xy[None, :, 0], a_xy[:, None, 1] - b_xy[None, :, 1])
    try:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(d)
        pairs = list(zip(r, c))
    except ImportError:
        pairs = _greedy_one_to_one(d)
    return [(int(i), int(j), float(d[i, j])) for i, j in pairs if d[i, j] <= max_dist]


def teammates_at_passes(links_m, cands_m, events=None, max_dist=10.0):
    """
    For one match: identify every freeze-frame teammate at each linked pass
    with a PFF player, by one-to-one nearest matching against PFF's snapshot
    of the passing team (passer excluded) at the same pass.

    links_m : rows of pff_pass_links.pkl for this match
    cands_m : rows of candidates_full.pkl for this match (any index; the index
              is kept as cand_index so other per-candidate tables can be joined)

    Returns one row per freeze-frame teammate (in_freeze_frame = True), with
    the PFF match if one lies within max_dist yards, plus one row per PFF
    teammate that no freeze-frame teammate was matched to
    (in_freeze_frame = False).
    """
    gid = int(links_m['pff_game_id'].iloc[0])
    ep = event_players(gid, events, set(links_m['pff_possession_event_id']))
    ep = ep[ep['passing_team']]
    by_event = {k: g for k, g in ep.groupby('possession_event_id')}
    lk = links_m.set_index('sb_event_id')
    cands_m = cands_m[cands_m['original_event_id'].isin(lk.index)]
    rows = []
    for eid, g in cands_m.groupby('original_event_id', sort=False):
        pe = lk.at[eid, 'pff_possession_event_id']
        passer = lk.at[eid, 'pff_passer_id']
        t = by_event.get(pe)
        t = t.iloc[0:0] if t is None else t[t['pff_player_id'] != passer]
        pairs = match_points(g[['cand_x', 'cand_y']].to_numpy(), t[['sb_x', 'sb_y']].to_numpy(), max_dist)
        hit = {i: (j, d) for i, j, d in pairs}
        used = {j for _, j, _ in pairs}
        base = {'match_id': int(links_m['match_id'].iloc[0]), 'original_event_id': eid,
                'pff_possession_event_id': pe}
        for i, (idx, r) in enumerate(g.iterrows()):
            row = dict(base, in_freeze_frame=True, cand_index=idx, cand_x=r['cand_x'],
                       cand_y=r['cand_y'], visible=bool(r['visible']), matched=i in hit)
            if i in hit:
                j, d = hit[i]
                p = t.iloc[j]
                row.update(pff_player_id=p['pff_player_id'], position_group=p['position_group'],
                           visibility=p['visibility'], confidence=p['confidence'], speed=p['speed'],
                           pff_x=p['sb_x'], pff_y=p['sb_y'], match_dist_yd=d)
            rows.append(row)
        for j in range(len(t)):
            if j in used:
                continue
            p = t.iloc[j]
            rows.append(dict(base, in_freeze_frame=False, matched=False,
                             pff_player_id=p['pff_player_id'], position_group=p['position_group'],
                             visibility=p['visibility'], confidence=p['confidence'], speed=p['speed'],
                             pff_x=p['sb_x'], pff_y=p['sb_y']))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- tracking frames around passes
import re as _re

_FRAME_RX = _re.compile(r'"frameNum":\s*(\d+)')


def sb_to_pff(sb_x, sb_y, direction, length=105.0, width=68.0):
    """Inverse of pff_to_sb: StatsBomb yards (attacking toward x = 120) -> PFF metres."""
    sb_x = np.asarray(sb_x, dtype=float)
    sb_y = np.asarray(sb_y, dtype=float)
    right = np.asarray(direction) == 'R'
    sx = SB_LENGTH / np.asarray(length, dtype=float)
    sy = SB_WIDTH / np.asarray(width, dtype=float)
    half_l = np.asarray(length, dtype=float) / 2
    half_w = np.asarray(width, dtype=float) / 2
    x = np.where(right, sb_x / sx - half_l, half_l - sb_x / sx)
    y = np.where(right, half_w - sb_y / sy, sb_y / sy - half_w)
    return x, y


def _players_array(lst):
    """Columns: jersey, x, y, visible (1 = VISIBLE, 0 = ESTIMATED)."""
    if not lst:
        return np.zeros((0, 4))
    return np.array([[int(p['jerseyNum']), p['x'], p['y'],
                      1.0 if p.get('visibility') == 'VISIBLE' else 0.0] for p in lst], dtype=float)


def extract_frames(gid, frame_nums):
    """
    Read only the wanted frames from a tracking file. Lines are skipped by
    their frame number before any JSON parsing.
    Returns {frameNum: {'home', 'away', 'home_s', 'away_s', 'ball'}} where the
    player entries are arrays from _players_array (raw and Kalman smoothed)
    and 'ball' is (x, y) or None.
    """
    keep = set(int(f) for f in frame_nums)
    out = {}
    with bz2.open(tracking_path(gid), 'rt') as fh:
        for line in fh:
            m = _FRAME_RX.search(line, 0, 400)
            if not m or int(m.group(1)) not in keep:
                continue
            d = json.loads(line)
            ball = (d.get('balls') or [None])[0]
            out[d['frameNum']] = {
                'home': _players_array(d.get('homePlayers')),
                'away': _players_array(d.get('awayPlayers')),
                'home_s': _players_array(d.get('homePlayersSmoothed')),
                'away_s': _players_array(d.get('awayPlayersSmoothed')),
                'ball': None if ball is None else (ball['x'], ball['y']),
            }
    return out


def pass_context(frames, f0, home_ball, lag=6, fps=29.97, max_speed=12.0):
    """
    Positions and velocities at the moment of a pass (frame f0).
    Positions are the raw tracked positions at f0. Velocities come from the
    Kalman-smoothed positions at f0 and f0 - lag (0.2 s at 30 fps), capped at
    max_speed m/s. Returns None if frame f0 is missing.
    Each of 'team' and 'opp' is a dict of arrays: jersey, pos (n, 2),
    vel (n, 2), visible (n,).
    """
    a = frames.get(f0)
    if a is None:
        return None
    b = frames.get(f0 - lag)
    out = {}
    for key, side in (('team', 'home' if home_ball else 'away'), ('opp', 'away' if home_ball else 'home')):
        raw = a[side]
        vel = np.zeros((len(raw), 2))
        if b is not None and len(a[side + '_s']) and len(b[side + '_s']):
            now, before = a[side + '_s'], b[side + '_s']
            for i, j in enumerate(raw[:, 0]):
                k1 = np.flatnonzero(now[:, 0] == j)
                k0 = np.flatnonzero(before[:, 0] == j)
                if len(k1) and len(k0):
                    vel[i] = (now[k1[0], 1:3] - before[k0[0], 1:3]) / (lag / fps)
        sp = np.hypot(vel[:, 0], vel[:, 1])
        vel = vel * np.minimum(1.0, max_speed / np.maximum(sp, 1e-9))[:, None]
        out[key] = {'jersey': raw[:, 0].astype(int), 'pos': raw[:, 1:3], 'vel': vel,
                    'visible': raw[:, 3].astype(bool)}
    out['ball'] = a['ball']
    return out


def time_to_reach(pos, vel, targets, reaction=0.7, vmax=7.0):
    """
    Simple player motion model (as in Spearman et al., 2017): the player keeps
    their current velocity during a reaction time, then runs straight to the
    target at vmax. pos, vel: (n, 2); targets: (m, 2). Returns (n, m) seconds.
    """
    pos = np.asarray(pos, dtype=float).reshape(-1, 2)
    vel = np.asarray(vel, dtype=float).reshape(-1, 2)
    targets = np.asarray(targets, dtype=float).reshape(-1, 2)
    p = pos + vel * reaction
    d = np.hypot(targets[None, :, 0] - p[:, None, 0], targets[None, :, 1] - p[:, None, 1])
    return reaction + d / vmax


def check_pass(start, target, ball_time, opp_pos, opp_vel, receiver_pos=None, receiver_vel=None,
               step=1.0, reaction=0.7, vmax=7.0):
    """
    Could a ground pass from start to target arrive?

    The ball travels along the straight line, reaching distance s after
    ball_time(s) seconds. path_margin is the smallest (defender time - ball
    time) over every defender and every point on the line (every `step`
    metres, target included). Negative means some defender can reach the line
    before the ball, so the pass would likely be cut out.

    If a receiver is given, they must also reach the target before any
    defender can: arrives = path_margin > 0 and receiver_time < defender time
    at the target.
    """
    start = np.asarray(start, dtype=float)
    target = np.asarray(target, dtype=float)
    d = float(np.hypot(*(target - start)))
    if d <= step:
        s = np.array([max(d, 1e-6)])
    else:
        s = np.append(np.arange(step, d, step), d)
    pts = start + (target - start) * (s / max(d, 1e-9))[:, None]
    tb = ball_time(s)
    out = {'dist_m': d, 'ball_time': float(tb[-1])}
    if len(opp_pos):
        td = time_to_reach(opp_pos, opp_vel, pts, reaction, vmax)
        m = td - tb[None, :]
        k = np.unravel_index(np.argmin(m), m.shape)
        out.update(path_margin=float(m[k]), decisive_defender=int(k[0]),
                   intercept_at_m=float(s[k[1]]), defender_time_at_target=float(td[:, -1].min()))
    else:
        out.update(path_margin=np.inf, decisive_defender=-1, intercept_at_m=np.nan,
                   defender_time_at_target=np.inf)
    if receiver_pos is not None:
        tr = float(time_to_reach(receiver_pos, receiver_vel, target[None], reaction, vmax)[0, 0])
        out['receiver_time'] = tr
        out['arrives'] = bool(out['path_margin'] > 0 and tr < out['defender_time_at_target'])
    return out


# ---------------------------------------------------------------- whole-match arrays (RQ3 with PFF)
def extract_match_arrays(gid, step=3):
    """
    Read a whole tracking file at 30 / step frames per second (step = 3 gives
    10 Hz) into compact arrays, identifying players through the roster.

    Returns a dict:
      frame     (T,) int     frameNum of each kept frame (multiples of step)
      period    (T,) int8
      player_id (P,) int     every PFF player seen in the file
      home      (P,) bool    True for the home team
      xy        (T, P, 2) float32, raw tracked positions in PFF metres, NaN when absent
      vis       (T, P) int8  1 VISIBLE, 0 ESTIMATED, -1 absent
      ball      (T, 3) float32 (x, y, z), NaN when missing
    """
    meta = json.load(open(metadata_path(gid), encoding='utf-8'))[0]
    home_name = meta['homeTeam']['name']
    ro = load_roster(gid)
    pid_of = {(r.team == home_name, r.jersey): r.pff_player_id for r in ro.itertuples()}
    frames, periods, rows, balls = [], [], [], []
    with bz2.open(tracking_path(gid), 'rt') as fh:
        for line in fh:
            m = _FRAME_RX.search(line, 0, 400)
            if not m or int(m.group(1)) % step:
                continue
            d = json.loads(line)
            frames.append(d['frameNum'])
            periods.append(d.get('period') or 0)
            entry = {}
            for is_home, key in ((True, 'homePlayers'), (False, 'awayPlayers')):
                for p in d.get(key) or []:
                    pid = pid_of.get((is_home, int(p['jerseyNum'])))
                    if pid is not None and pid not in entry:
                        entry[pid] = (p['x'], p['y'], 1 if p.get('visibility') == 'VISIBLE' else 0, is_home)
            rows.append(entry)
            b = (d.get('balls') or [None])[0]
            balls.append((b['x'], b['y'], b.get('z') or 0.0) if b else (np.nan, np.nan, np.nan))
    ids = sorted({pid for e in rows for pid in e})
    col = {pid: i for i, pid in enumerate(ids)}
    home = np.zeros(len(ids), dtype=bool)
    xy = np.full((len(rows), len(ids), 2), np.nan, dtype=np.float32)
    vis = np.full((len(rows), len(ids)), -1, dtype=np.int8)
    for t, e in enumerate(rows):
        for pid, (x, y, v, h) in e.items():
            j = col[pid]
            xy[t, j] = (x, y)
            vis[t, j] = v
            home[j] = h
    return {'frame': np.array(frames, dtype=np.int64), 'period': np.array(periods, dtype=np.int8),
            'player_id': np.array(ids, dtype=np.int64), 'home': home, 'xy': xy, 'vis': vis,
            'ball': np.array(balls, dtype=np.float32)}


def headings(xy, frame=None, step=3, lag=5, min_speed=1.0, dt=0.1, hold=20):
    """
    Movement direction of every player at every 10 Hz sample, used as the
    facing direction for the vision cone (the same idea as the StatsBomb
    version, which uses the movement into the pass).

    xy: (T, P, 2); frame: (T,) frame numbers, used to skip gaps. Heading at t
    is the direction from t - lag to t (0.5 s at 10 Hz). If the player moves slower than min_speed m/s, the last valid
    heading from up to `hold` samples (2 s) earlier is kept; otherwise NaN.
    Returns unit vectors (T, P, 2) and speeds (T, P).
    """
    T = xy.shape[0]
    d = np.full_like(xy, np.nan)
    d[lag:] = xy[lag:] - xy[:-lag]
    if frame is not None:
        # no heading across a gap in the tracking (replays, stoppages, half time)
        gap = np.ones(T, dtype=bool)
        gap[lag:] = (frame[lag:] - frame[:-lag]) != lag * step
        d[gap] = np.nan
    speed = np.hypot(d[..., 0], d[..., 1]) / (lag * dt)
    unit = d / np.maximum(np.hypot(d[..., 0], d[..., 1]), 1e-9)[..., None]
    ok = speed >= min_speed
    head = np.where(ok[..., None], unit, np.nan)
    last = np.full(xy.shape[1:], np.nan, dtype=xy.dtype)
    age = np.full(xy.shape[1], 10 ** 6)
    out = np.full_like(xy, np.nan)
    for t in range(T):
        good = ok[t]
        last[good] = head[t, good]
        age[good] = 0
        age[~good] += 1
        keep = age <= hold
        out[t, keep] = last[keep]
    return out, speed
