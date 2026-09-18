"""Read-only queries for the saved election simulations. No model API calls."""
from __future__ import annotations

import json
import sqlite3
import zlib
import sys
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

APP_DIR = Path(__file__).resolve().parent
ROOT = next((p for p in APP_DIR.parents if (p / 'thesis_utils.py').exists()), APP_DIR.parents[1])
sys.path.insert(0, str(ROOT))
import thesis_utils as tu

DB = APP_DIR / 'generated' / 'explorer.sqlite'
HF_DATASET = 'Jorge18tu/grounding-the-electorate-data'  # public copy of the index with the answers embedded


def ensure_index():
    """Streamlit Cloud has no local checkpoints: download the 2 GB index once from the dataset repo."""
    if DB.exists():
        return
    from huggingface_hub import hf_hub_download
    DB.parent.mkdir(parents=True, exist_ok=True)
    for name in ('explorer.sqlite', 'sv_census.json'):
        hf_hub_download(HF_DATASET, name, repo_type='dataset', local_dir=str(DB.parent))


ensure_index()
COUNTRIES = {
    'usa': dict(label='USA 2024', candidates=tu.USA_CANDIDATES, colors=['#C9252D', '#246BCE'], default='PA', geo='us_states.geojson', key='code', region='state'),
    'el_salvador': dict(label='El Salvador 2024', candidates=tu.SV_CANDIDATES, colors=['#00A6CF', '#D62828', '#234B8A'], default='San Salvador', geo='el_salvador_departments.geojson', key='shapeName', region='department'),
    'brazil': dict(label='Brazil 2026', candidates=tu.BR_CANDIDATES, colors=['#C8102E', '#0F0073'], default='São Paulo', geo='brazil_states.geojson', key='name', region='region'),
}
ARMS = {'demographic': 'Demographic', 'cultural': 'Cultural Background', 'persona': 'Persona', 'career': 'Career + Big Five'}


def registry():
    runs = []
    for key, (label, path) in tu.USA_CHECKPOINTS.items():
        arm = {'demo': 'demographic', 'bg': 'cultural', 'pers': 'persona', 'career': 'career'}[key]
        runs.append(dict(id=f'usa_{arm}_en', country='usa', arm=arm, language='en', label=label+' · EN', path=str(path.relative_to(ROOT))))
    for (arm, lang), path in tu.SV_CHECKPOINTS.items():
        runs.append(dict(id=f'el_salvador_{arm}_{lang}', country='el_salvador', arm=arm, language=lang, label=ARMS[arm]+' · '+lang.upper(), path=str(path.relative_to(ROOT))))
    for (context, arm, lang), path in tu.BR_CHECKPOINTS.items():
        if context == 'short':
            runs.append(dict(id=f'brazil_{arm}_{lang}', country='brazil', arm=arm, language=lang, label=ARMS[arm]+' · '+lang.upper(), path=str(path.relative_to(ROOT))))
    return runs


RUNS = {r['id']: r for r in registry()}


def connection():
    return sqlite3.connect(DB.resolve().as_uri()+'?mode=ro', uri=True)


def query(sql, params=()):
    with closing(connection()) as db:
        return pd.read_sql_query(sql, db, params=params)


def run_frame(run):
    return query('SELECT r.pid, r.valid, r.winner, r.p0, r.p1, r.p2, p.region, p.city, p.age, p.sex, p.education FROM responses r JOIN people p ON r.pid=p.pid WHERE r.run=?', (run,))


def aggregates(frame, country, metric='probability'):
    valid = frame.loc[frame.valid.eq(1)].copy()
    n = len(COUNTRIES[country]['candidates'])
    cols = [f'p{i}' for i in range(n)]
    if metric == 'choice':
        for i, col in enumerate(cols):
            valid[col] = valid.winner.eq(i).astype(float)*100
    grouped = valid.groupby('region')[cols].mean()
    grouped['valid_n'] = valid.groupby('region').size()
    grouped['total_n'] = frame.groupby('region').size()
    grouped['ties'] = valid.groupby('region').winner.apply(lambda x: int(x.eq(-1).sum()))
    # Retain territories containing only invalid responses as missing, not zero support.
    grouped = grouped.reindex(sorted(frame.region.unique()))
    grouped['total_n'] = frame.groupby('region').size()
    grouped[['valid_n','ties']] = grouped[['valid_n','ties']].fillna(0)
    if country == 'usa':
        w = pd.Series(tu.POPULATION_2025).reindex(grouped.index).where(grouped.valid_n.gt(0))
        national = grouped[cols].mul(w, axis=0).sum()/w.sum() if w.sum() else pd.Series(np.nan, index=cols)
    else:
        national = valid[cols].mean()
    return grouped, national


def matched(a, b):
    return a.loc[a.valid.eq(1)].merge(b.loc[b.valid.eq(1)], on='pid', suffixes=('_a','_b'), validate='one_to_one')


def electoral_college(summary):
    """Assign 2024 electors winner-take-all by simulated state leader, as thesis_utils does.

    Maine and Nebraska give all their electors (4 and 5) to the statewide leader:
    the data have no district results, and the thesis notebooks use the same rule.
    Missing states and tied state aggregates stay unassigned.
    """
    totals = [0, 0]
    unassigned = 0
    for state, votes in tu.EV_2024.items():
        if state not in summary.index or summary.loc[state, 'valid_n'] <= 0:
            unassigned += votes
            continue
        values = summary.loc[state, ['p0', 'p1']].to_numpy(dtype=float)
        if not np.isfinite(values).all() or abs(values[0] - values[1]) < 1e-8:
            unassigned += votes
        else:
            totals[int(values.argmax())] += votes
    winner = next((i for i, count in enumerate(totals) if count >= 270), None)
    return dict(totals=totals, unassigned=unassigned, winner=winner)


def pair_frames(pair):
    return tuple(pair[['pid']+[c for c in pair if c.endswith('_'+s)]].rename(columns=lambda c: c[:-2] if c.endswith('_'+s) else c) for s in ('a','b'))


def geometry(country):
    from shapely.geometry import shape, mapping
    from shapely.geometry.polygon import orient
    cfg = COUNTRIES[country]
    data = json.loads((ROOT/'data'/cfg['geo']).read_text(encoding='utf-8'))
    for f in data['features']:
        name = str(f['properties'][cfg['key']]).removeprefix('Departamento de ')
        geom = shape(f['geometry']).simplify(.025 if country != 'el_salvador' else .003, preserve_topology=True)
        parts = list(geom.geoms) if geom.geom_type == 'MultiPolygon' else [geom]
        parts = [mapping(orient(p, sign=-1)) for p in parts]
        f['geometry'] = parts[0] if len(parts)==1 else {'type':'MultiPolygon','coordinates':[p['coordinates'] for p in parts]}
        f['properties'] = {'id': name, 'name': tu.STATE_NAME.get(name, name) if country=='usa' else name}
    return data


def detail(pid, run):
    person = query('SELECT profile FROM people WHERE pid=?', (pid,))
    row = query('SELECT * FROM responses WHERE pid=? AND run=?', (pid,run))
    if person.empty or row.empty:
        return None
    record = row.iloc[0].to_dict()
    if record.get('raw') is not None:  # cloud bundle: the minimal answer record is stored in the index
        raw = json.loads(zlib.decompress(record['raw']))
    else:
        source = ROOT / RUNS[run]['path']
        meta = query('SELECT size, mtime FROM runs WHERE id=?', (run,)).iloc[0]
        if source.stat().st_size != meta['size'] or source.stat().st_mtime_ns != int(meta['mtime']):
            raise ValueError('Checkpoint changed after indexing. Rebuild the explorer data.')
        with source.open('rb') as handle:
            handle.seek(int(record['offset']))
            raw = json.loads(handle.readline())
    profile=json.loads(person.iloc[0]['profile'])
    run_cfg=RUNS[run]
    cfg=COUNTRIES[run_cfg['country']]
    response=raw.get('response') or {}
    normalized={
        'schema_version':'2.0','run_id':run,'response_id':run+':'+pid,'persona_id':pid,
        'country':run_cfg['country'],'valid':bool(record['valid']),
        'territory':{'level':'department' if run_cfg['country']=='el_salvador' else 'state','name':profile.get(cfg['region']),'locality':profile.get('city',profile.get('municipality'))},
        'experiment':{'condition':run_cfg['arm'],'language':run_cfg['language'],'prompt_name':raw.get('prompt_name')},
        'result':{'probabilities':{f'candidate_{i}':response.get(c) for i,c in enumerate(cfg['candidates'])},'candidate_labels':{f'candidate_{i}':c for i,c in enumerate(cfg['candidates'])},'winner_id':f'candidate_{record["winner"]}' if record['valid'] and record['winner']>=0 else None,'reason':next((response[k] for k in ('reason','razon','razao') if k in response),None)},
        'provenance':{k:raw.get(k) for k in ['model','provider','generation_config','status','in_tok','out_tok','latency_s','attempts']},
        'prompt':{'exact_submitted_prompt':None,'availability':'not stored; current context shown separately'},
        'source':{'file':RUNS[run]['path'],'record_id':raw.get('_id'),'byte_offset':int(record['offset'])},
    }
    return dict(profile=profile, raw=raw, normalized=normalized)


def current_context(run):
    r = RUNS[run]
    lang, country = r['language'], r['country']
    ctx = tu.PROMPTS[country]
    if country == 'usa':
        return ctx
    return {k:v for k,v in ctx.items() if k.endswith('_'+lang) and ('full' not in k)}
