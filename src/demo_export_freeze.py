"""
Export the StatsBomb 360 freeze-frame counterfactuals for all 166 matches
for the pass explorer demo. One file per match, data/demo/freeze_{match_id}.json.gz,
plus data/demo/index.json listing every match.

Every position is in StatsBomb yards with the passing team attacking toward
x = 120, as in the events and freeze frames.

Run from the notebooks folder:  python ../src/demo_export_freeze.py
"""
import gzip, json, os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from vision_cone import add_orientation_vectors

RESULTS = '../data/results'
CACHE = '../data/statsbomb_cache'
OUT = '../data/demo'
LIMIT = float(os.environ.get('DEMO_LIMIT', 1e9))   # seconds per run (to split long runs)


def r(v, nd=1):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return round(float(v), nd)


def b(v):
    if v is None or pd.isna(v):
        return None
    return 1 if bool(v) else 0


def main():
    T0 = time.time()
    pl = pd.read_pickle(f'{RESULTS}/rq2_risk_context_pass_level.pkl')
    pl['match_id'] = pl['match_id'].astype(str)
    cands = pd.read_pickle(f'{RESULTS}/candidates_full.pkl')
    cands['match_id'] = cands['match_id'].astype(str)
    cf = pd.read_pickle(f'{RESULTS}/rq3_crossfit_predictions.pkl')
    cf['key'] = list(zip(cf['original_event_id'], cf['cand_x'].round(2), cf['cand_y'].round(2)))
    mem = cf.set_index('key')[['err_lstm_m', 'n_hist']]
    mem = mem[~mem.index.duplicated()]
    links = pd.read_pickle(f'{RESULTS}/pff_pass_links.pkl')
    pff_of = dict(zip(links['sb_event_id'], links['pff_game_id']))
    pe_of = dict(zip(links['sb_event_id'], links['pff_possession_event_id']))
    index = []
    by_match = cands.groupby('match_id')
    for mid, pm in pl.groupby('match_id'):
        path = os.path.join(OUT, f'freeze_{mid}.json.gz')
        if os.path.exists(path):
            doc = json.load(gzip.open(path))
            passes = doc['passes']
            index.append({'match_id': mid, 'competition': doc['competition'], 'teams': doc['teams'], 'n': len(passes),
                          'flagged': int(sum(q['rq2'][0] == 1 for q in passes)),
                          'pff': int(next((q['pff'][0] for q in passes if q['pff']), 0)) or None})
            continue
        if time.time() - T0 > LIMIT:
            print('time limit reached; run again to continue')
            return
        ev = add_orientation_vectors(pd.read_pickle(f'{CACHE}/events_{mid}.pkl')).set_index('id')
        fr = pd.read_pickle(f'{CACHE}/frames_{mid}.pkl')
        fr_g = {k: g for k, g in fr.groupby('id')}
        cm = by_match.get_group(mid).groupby('original_event_id')
        teams = sorted(ev['team'].dropna().unique().tolist())
        comp = pm['competition'].iloc[0]
        passes = []
        for p in pm.sort_values(['period', 'minute']).itertuples():
            eid = p.original_event_id
            if eid not in ev.index or eid not in cm.groups:
                continue
            e = ev.loc[eid]
            loc, end = e['location'], e['pass_end_location']
            ori = e['orientation_vector']
            c = cm.get_group(eid)
            mates = []
            d_end = np.hypot(c['cand_x'] - end[0], c['cand_y'] - end[1])
            rec_i = d_end.idxmin() if len(c) and d_end.min() <= 5 else None
            for i, t in c.iterrows():
                mm = mem.loc[[(eid, round(t['cand_x'], 2), round(t['cand_y'], 2))]].iloc[0] \
                    if (eid, round(t['cand_x'], 2), round(t['cand_y'], 2)) in mem.index else None
                mates.append([r(t['cand_x']), r(t['cand_y']), b(t['visible']), r(t['candidate_value'], 4),
                              r(t['xp_value'], 3), b(t['offside']), b(t['maximal_vaep']),
                              b(t['legitimate_maximal_vaep']),
                              None if mm is None else r(mm['err_lstm_m'], 2),
                              None if mm is None else int(mm['n_hist']), 1 if i == rec_i else 0])
            f = fr_g.get(eid)
            opp, area = [], None
            if f is not None:
                for q in f.itertuples():
                    if not q.teammate:
                        opp.append([r(q.location[0]), r(q.location[1]), 1 if q.keeper else 0])
                va = f['visible_area'].iloc[0]
                if isinstance(va, (list, np.ndarray)) and len(va) >= 6:
                    area = [r(v) for v in va]
            passes.append({
                'e': eid, 'min': int(e['minute']), 'sec': int(e['second']), 'per': int(e['period']),
                'team': e['team'], 'pl': e['player'],
                'to': None if pd.isna(e.get('pass_recipient')) else e['pass_recipient'],
                'P': [r(loc[0]), r(loc[1])], 'E': [r(end[0]), r(end[1])],
                'o': None if ori is None or (isinstance(ori, float) and pd.isna(ori)) else [r(ori[0], 3), r(ori[1], 3)],
                'rv': r(c['real_value'].iloc[0], 4), 'rxp': r(p.real_xp, 3),
                'rq2': [b(p.visible_orig), b(p.visible_ra), b(p.all_orig), b(p.visible_strict)],
                'press': b(p.under_pressure),
                'pff': None if eid not in pff_of else [int(pff_of[eid]), int(pe_of[eid])],
                'm': mates, 'x': opp, 'va': area})
        doc = {'match_id': mid, 'competition': comp, 'teams': teams, 'passes': passes}
        raw = json.dumps(doc, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
        with gzip.open(path, 'wb', compresslevel=9) as fh:
            fh.write(raw)
        index.append({'match_id': mid, 'competition': comp, 'teams': teams, 'n': len(passes),
                      'flagged': int(sum(q['rq2'][0] == 1 for q in passes)),
                      'pff': int(next((q['pff'][0] for q in passes if q['pff']), 0)) or None})
        print(mid, comp, teams, len(passes), len(raw), flush=True)
    json.dump(index, open(os.path.join(OUT, 'freeze_index.json'), 'w'), ensure_ascii=False)


if __name__ == '__main__':
    main()
