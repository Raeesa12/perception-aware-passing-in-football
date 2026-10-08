"""
Pass reachability from PFF tracking (supervisor point 1b).

For each linked World Cup pass, the PFF tracking frame at the moment of the
pass gives every player's position and velocity. A ground pass from the
passer to a target point "arrives" when
  1. no defender can reach any point of the ball's straight path before the
     ball does (pff.check_pass, path_margin > 0), and
  2. the intended teammate can reach the target before any defender can.

Player motion follows the simple model of Spearman et al. (2017): keep the
current velocity for a reaction time, then run straight at a maximum speed.
The ball travels at a constant speed v.

process_match returns three tables for one match:
  real    : every linked pass (completed or not), the path margin of the
            real pass at each ball speed, for validating the model
  mates   : every freeze-frame teammate at an analysed pass, whether a pass
            to (a) their true PFF position and (b) their freeze-frame position
            would arrive
  unseen  : the teammates outside the cone that have an LSTM prediction,
            whether a pass to the LSTM's predicted position would arrive
"""
import numpy as np
import pandas as pd

import pff

SPEEDS = (10.0, 12.5, 15.0, 20.0)                 # ball speeds, m/s
MAIN_SPEED = 12.5
REACTION, VMAX = 0.7, 7.0
SENSITIVITY = {'react0.5': (0.5, 7.0), 'react0.9': (0.9, 7.0),
               'vmax5': (0.7, 5.0), 'vmax9': (0.7, 9.0)}
LAG = 6                                           # frames for velocity (0.2 s)


def _ball(v):
    return lambda s: s / v


def process_match(mid, links_m, pff_passes_g, sb_events, tm_m, lstm_m, analysed_ids, roster, fps):
    gid = int(links_m['pff_game_id'].iloc[0])
    L = links_m.join(pff_passes_g.set_index('possession_event_id')[['event_time', 'home_team', 'attacking_direction']],
                     on='pff_possession_event_id')
    L = L.join(sb_events[['pass_end_location', 'pass_outcome', 'pass_height']], on='sb_event_id')
    L = L[L['attacking_direction'].isin(['R', 'L'])].copy()
    L['analysed'] = L['sb_event_id'].isin(analysed_ids)
    L['f0'] = (L['event_time'] * fps).round().astype(int)
    frames = pff.extract_frames(gid, set(L['f0']) | set(L['f0'] - LAG))

    jersey_of = dict(zip(roster['pff_player_id'], roster['jersey']))
    tm_by = {k: g for k, g in tm_m[tm_m['in_freeze_frame'] & tm_m['matched']].groupby('original_event_id')}
    lstm_by = {k: g.set_index('cand_index') for k, g in lstm_m.groupby('original_event_id')}
    real, mates, unseen = [], [], []

    for r in L.itertuples():
        ctx = pff.pass_context(frames, r.f0, r.home_team, lag=LAG, fps=fps)
        if ctx is None:
            continue
        team, opp = ctx['team'], ctx['opp']
        pj = jersey_of.get(int(r.pff_passer_id)) if pd.notna(r.pff_passer_id) else None
        k = np.flatnonzero(team['jersey'] == pj) if pj is not None else []
        if len(k):
            start = team['pos'][k[0]]
        elif ctx['ball'] is not None:
            start = np.array(ctx['ball'], dtype=float)
        else:
            continue

        def to_pff(x, y):
            return np.array(pff.sb_to_pff(x, y, r.attacking_direction), dtype=float)

        # 1. the real pass, for validation
        if isinstance(r.pass_end_location, list):
            tgt = to_pff(*r.pass_end_location[:2])
            row = {'match_id': mid, 'original_event_id': r.sb_event_id, 'analysed': r.analysed,
                   'complete': pd.isna(r.pass_outcome), 'height': r.pass_height}
            for v in SPEEDS:
                c = pff.check_pass(start, tgt, _ball(v), opp['pos'], opp['vel'], reaction=REACTION, vmax=VMAX)
                row[f'margin_v{v:g}'] = c['path_margin']
                if v == MAIN_SPEED:
                    row['dist_m'] = c['dist_m']
                    d = c['decisive_defender']
                    row['decisive_defender_visible'] = bool(opp['visible'][d]) if d >= 0 else np.nan
            real.append(row)

        if not r.analysed:
            continue
        tmg = tm_by.get(r.sb_event_id)
        if tmg is None:
            continue
        lsg = lstm_by.get(r.sb_event_id)

        for t in tmg.itertuples():
            j = jersey_of.get(int(t.pff_player_id)) if pd.notna(t.pff_player_id) else None
            kk = np.flatnonzero(team['jersey'] == j) if j is not None else []
            if not len(kk):
                continue
            pos, vel = team['pos'][kk[0]], team['vel'][kk[0]]
            base = {'match_id': mid, 'original_event_id': r.sb_event_id, 'cand_index': t.cand_index,
                    'visible_cone': t.visible, 'mate_visible_pff': bool(team['visible'][kk[0]])}
            targets = {'true': pos, '360': to_pff(t.cand_x, t.cand_y)}
            row = dict(base)
            for name, tgt in targets.items():
                for v in SPEEDS:
                    c = pff.check_pass(start, tgt, _ball(v), opp['pos'], opp['vel'], pos, vel,
                                       reaction=REACTION, vmax=VMAX)
                    row[f'arr_{name}_v{v:g}'] = c['arrives']
                    if v == MAIN_SPEED:
                        row[f'path_{name}'] = c['path_margin'] > 0
            mates.append(row)

            if lsg is not None and t.cand_index in lsg.index:
                l = lsg.loc[t.cand_index]
                pred = to_pff(l['lstm_x'], l['lstm_y'])
                u = dict(base, err_lstm_pff=l['err_lstm_pff'], err_lstm_360=l['err_lstm_360'],
                         n_hist=l['n_hist'], horizon_s=l['horizon_s'])
                for name, tgt in (('lstm', pred), ('true', pos)):
                    for v in SPEEDS:
                        c = pff.check_pass(start, tgt, _ball(v), opp['pos'], opp['vel'], pos, vel,
                                           reaction=REACTION, vmax=VMAX)
                        u[f'arr_{name}_v{v:g}'] = c['arrives']
                        if v == MAIN_SPEED:
                            u[f'path_{name}'] = c['path_margin'] > 0
                            u[f'mate_late_{name}'] = c['receiver_time'] >= c['defender_time_at_target']
                            u[f'dist_{name}_m'] = c['dist_m']
                            if name == 'lstm':
                                d = c['decisive_defender']
                                u['decisive_defender_visible'] = bool(opp['visible'][d]) if d >= 0 else np.nan
                    for label, (re_, vm) in SENSITIVITY.items():
                        c = pff.check_pass(start, tgt, _ball(MAIN_SPEED), opp['pos'], opp['vel'], pos, vel,
                                           reaction=re_, vmax=vm)
                        u[f'arr_{name}_{label}'] = c['arrives']
                unseen.append(u)

    return pd.DataFrame(real), pd.DataFrame(mates), pd.DataFrame(unseen)
