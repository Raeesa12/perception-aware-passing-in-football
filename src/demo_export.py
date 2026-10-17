"""
Export the data behind the pass explorer demo (World Cup 2022, PFF tracking).

For every game it writes two files to data/demo/:
  tracking_{gid}.bin.gz  every player and the ball at 5 Hz, positions in
                         decimetres (int16), delta-encoded over time, gzipped
  passes_{gid}.json.gz   every PFF pass with the passer's cone, each
                         teammate's memory (seen in the last 5 s, LSTM
                         position) and, where the pass is in RQ2, the value
                         of the hypothetical pass to each teammate

Binary layout (little endian), after gunzip:
  int32 T, int32 P
  int32 frame[T]            PFF frame number of each sample
  int8  period[T]
  int16 xy[P][2][T]         player-major, x series then y series, decimetres;
                            first value absolute, then deltas from the
                            previous valid value; -32768 = absent
  int8  vis[P][T]           1 visible, 0 estimated, -1 absent
  int16 ball[3][T]          x, y, z in decimetres, same delta coding

Run from the notebooks folder:  python ../src/demo_export.py [gid ...]
"""
import gzip, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import pff

RESULTS = '../data/results'
ARR_DIR = '../data/pff_arrays'
OUT = '../data/demo'
NA = -32768
SUB = 2   # keep every 2nd 10 Hz sample -> 5 Hz


def _delta(series_dm):
    """series_dm: float array in decimetres with NaN; returns int16 delta code."""
    q = np.round(series_dm)
    valid = ~np.isnan(q)
    prev = pd.Series(np.where(valid, q, np.nan)).ffill().shift(1).fillna(0).to_numpy()
    out = np.full(len(q), NA, dtype=np.int16)
    out[valid] = np.clip(q[valid] - prev[valid], -32767, 32767).astype(np.int16)
    return out


def write_tracking(gid, arr):
    keep = np.arange(0, len(arr['frame']), SUB)
    xy = arr['xy'][keep] * 10.0
    ball = arr['ball'][keep] * 10.0
    T, P = xy.shape[0], xy.shape[1]
    parts = [np.array([T, P], dtype='<i4').tobytes(),
             arr['frame'][keep].astype('<i4').tobytes(),
             arr['period'][keep].astype('i1').tobytes()]
    for j in range(P):
        for k in range(2):
            parts.append(_delta(xy[:, j, k]).astype('<i2').tobytes())
    parts.append(np.ascontiguousarray(arr['vis'][keep].T).astype('i1').tobytes())
    for k in range(3):
        parts.append(_delta(ball[:, k]).astype('<i2').tobytes())
    raw = b''.join(parts)
    with gzip.open(os.path.join(OUT, f'tracking_{gid}.bin.gz'), 'wb', compresslevel=9) as fh:
        fh.write(raw)
    return len(raw)


def r3(v, nd=3):
    if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v):
        return None
    return round(float(v), nd)


def b(v):
    if v is None or pd.isna(v):
        return None
    return 1 if bool(v) else 0


def load_shared():
    S = {}
    S['meta'] = pff.load_metadata().set_index('pff_game_id')
    S['passes'] = pd.read_pickle(f'{RESULTS}/pff_passes_all.pkl')
    S['mates'] = pd.read_pickle(f'{RESULTS}/pff_memory_mates.pkl')
    pred = pd.read_pickle(f'{RESULTS}/pff_lstm_predictions.pkl')
    S['pred'] = pred.set_index(['possession_event_id', 'mate_id'])[['pred_lstm_x', 'pred_lstm_y', 'err_lstm']]
    links = pd.read_pickle(f'{RESULTS}/pff_pass_links.pkl')
    S['links'] = links.set_index('pff_possession_event_id')['sb_event_id']
    pl = pd.read_pickle(f'{RESULTS}/rq2_risk_context_pass_level.pkl')
    S['pl'] = pl.set_index('original_event_id')[['visible_orig', 'visible_ra', 'all_orig', 'real_xp']]
    tm = pd.read_pickle(f'{RESULTS}/pff_teammates_at_passes.pkl')
    tm = tm[tm['in_freeze_frame'] & tm['matched'] & tm['cand_index'].notna()]
    cands = pd.read_pickle(f'{RESULTS}/candidates_full.pkl')
    reach = pd.read_pickle(f'{RESULTS}/rq3_reach_teammates.pkl').set_index('cand_index')['arr_true_v12.5']
    c = cands.loc[tm['cand_index'].astype(int), ['candidate_value', 'real_value', 'xp_value', 'offside',
                                                    'forward_progressive', 'legitimate_forward_progressive', 'visible']]
    c.index = pd.MultiIndex.from_arrays([tm['pff_possession_event_id'].to_numpy(), tm['pff_player_id'].astype(int).to_numpy()])
    c['arrive'] = reach.reindex(tm['cand_index'].to_numpy()).to_numpy()
    S['cf'] = c[~c.index.duplicated()]
    S['real_value'] = cands.groupby('original_event_id')['real_value'].first()
    S['tau'] = json.load(open(f'{RESULTS}/pff_tau.json'))['tau_m']
    return S


def write_passes(gid, arr, S):
    meta = S['meta'].loc[gid]
    fps = float(meta['fps'])
    ro = pff.load_roster(gid)
    frame = arr['frame']
    keep = np.arange(0, len(frame), SUB)
    head, _ = pff.headings(arr['xy'], frame=frame, step=3)
    pid_col = {int(p): j for j, p in enumerate(arr['player_id'])}
    # start frame of each period, for the match clock
    pstart = {int(p): int(frame[arr['period'] == p][0]) for p in np.unique(arr['period']) if p > 0}
    offset = {1: 0, 2: 45, 3: 90, 4: 105}
    mates_g = S['mates'][S['mates']['pff_game_id'] == gid].groupby('possession_event_id')
    passes_g = S['passes'][S['passes']['pff_game_id'] == gid].set_index('possession_event_id')
    out = []
    for pe, mg in mates_g:
        r = passes_g.loc[pe]
        f0 = int(round(r['event_time'] * fps))
        t0 = int(np.argmin(np.abs(frame - f0)))
        per = int(arr['period'][t0])
        secs = (frame[t0] - pstart.get(per, frame[t0])) / fps + 60 * offset.get(per, 0)
        jp = pid_col[int(r['passer_id'])]
        h = head[t0, jp]
        sign = 1 if r['attacking_direction'] == 'R' else -1
        sb_id = S['links'].get(pe)
        rec = {'e': int(pe), 't': round(t0 / SUB, 2), 'per': per, 'clk': int(secs),
               'p': int(r['passer_id']),
               'r': None if pd.isna(r['receiver_id']) else int(r['receiver_id']),
               'tg': None if pd.isna(r['target_id']) else int(r['target_id']),
               'o': r['outcome'], 'k': r['event_type'], 's': sign,
               'h': None if np.isnan(h[0]) else [r3(h[0]), r3(h[1])],
               'lab': None if pd.isna(r['better_option_player_id']) else int(r['better_option_player_id']),
               'lt': r['better_option_type']}
        if sb_id is not None and sb_id in S['pl'].index:
            f = S['pl'].loc[sb_id]
            rec['sb'] = sb_id
            rec['rq2'] = [b(f['visible_orig']), b(f['visible_ra']), b(f['all_orig'])]
            rec['rv'] = r3(S['real_value'].get(sb_id), 4)
            rec['rxp'] = r3(f['real_xp'])
        m = []
        for row in mg.itertuples():
            key = (pe, int(row.mate_id))
            pr = S['pred'].loc[key] if key in S['pred'].index else None
            cf = S['cf'].loc[key] if key in S['cf'].index else None
            gx, gy = (row.last_x, row.last_y) if pr is None else (pr['pred_lstm_x'], pr['pred_lstm_y'])
            m.append([int(row.mate_id), b(row.in_cone), None if pd.isna(row.n_seen) else int(row.n_seen),
                      r3(row.gap_s, 2),
                      # memory position in raw PFF coordinates (attack frame * sign)
                      None if pd.isna(gx) else r3(sign * gx, 2), None if pd.isna(gy) else r3(sign * gy, 2),
                      0 if pr is None else 1, None if pr is None else r3(pr['err_lstm'], 2),
                      None if cf is None else r3(cf['candidate_value'], 4),
                      None if cf is None else r3(cf['xp_value']),
                      None if cf is None else b(cf['offside']),
                      None if cf is None else b(cf['forward_progressive']),
                      None if cf is None else b(cf['legitimate_forward_progressive']),
                      None if cf is None else b(cf['arrive']),
                      None if cf is None else b(cf['visible'])])
        rec['m'] = m
        out.append(rec)
    out.sort(key=lambda d: d['t'])
    players = [{'id': int(p.pff_player_id), 'n': p.player_name, 'j': int(p.jersey), 'pos': p.position_group,
                'home': bool(p.team == meta['home_team'])} for p in ro.itertuples() if int(p.pff_player_id) in pid_col]
    doc = {'gid': int(gid), 'home': meta['home_team'], 'away': meta['away_team'], 'date': str(meta['date'])[:10],
           'fps': fps, 'sub': SUB, 'tau': round(S['tau'], 2), 'players': players,
           'cols': [int(p) for p in arr['player_id']], 'passes': out}
    raw = json.dumps(doc, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    with gzip.open(os.path.join(OUT, f'passes_{gid}.json.gz'), 'wb', compresslevel=9) as fh:
        fh.write(raw)
    return len(out), len(raw)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    S = load_shared()
    gids = [int(g) for g in sys.argv[1:]] or list(S['meta'].index)
    for gid in gids:
        if os.path.exists(os.path.join(OUT, f'passes_{gid}.json.gz')) and len(sys.argv) == 1:
            continue
        arr = dict(np.load(os.path.join(ARR_DIR, f'arrays_{gid}.npz')))
        n_raw = write_tracking(gid, arr)
        n_p, n_json = write_passes(gid, arr, S)
        print(gid, 'tracking', n_raw, 'passes', n_p, 'json', n_json, flush=True)
