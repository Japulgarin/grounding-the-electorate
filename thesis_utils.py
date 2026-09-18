"""Shared constants, checkpoint loaders and figures for the thesis notebooks.

Every notebook in this folder imports from here, so the analysis code lives in
one place. Nothing in this module sends a request to a model: it only reads the
`.jsonl` checkpoints written by `personasurvey`.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
CACHE_DIR = RESULTS_DIR / ".cache"
PROMPTS = json.loads((ROOT / "prompts" / "election_contexts.json").read_text(encoding="utf-8"))

MODEL_NAME = "openai/gpt-oss-20b"
SIMULATION_KWARGS = dict(
    endpoint_strategy="fixed", json_mode=False, max_tokens=500, temperature=0,
    reasoning_effort="low", retries=3, retry_delay=2, timeout=600,
)
REASON_FIELDS = ("reason", "razon", "razao")
PER_POD_CONCURRENCY = 600


def runpod_provider(pod_urls, model=MODEL_NAME, per_pod_concurrency=PER_POD_CONCURRENCY):
    """OpenAI-compatible vLLM pods on RunPod serving gpt-oss-20b."""
    from personasurvey import RunPodProvider
    return RunPodProvider(urls=pod_urls, model=model, api_key="EMPTY", concurrency=per_pod_concurrency)


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

# Wall-clock runtime and first-pass errors printed by personasurvey when each run finished
# (copied from the run logs in the original notebooks; not stored in the checkpoints).
RUN_LOG = {
    "25_AGUST_BACKGROUND.jsonl": {"runtime_min": 140.3, "pods": 2, "first_pass_errors": 0},
    "25_AGUST_PERSONA.jsonl": {"runtime_min": 140.0, "pods": 2, "first_pass_errors": 0},
    "25_AGUST_CAREER_OCEAN.jsonl": {"runtime_min": 171.4, "pods": 2, "first_pass_errors": 417},
    "el_salvador_500_english.jsonl": {"runtime_min": 1.1, "pods": 1, "first_pass_errors": 0},
    "el_salvador_500_spanish.jsonl": {"runtime_min": 1.3, "pods": 1, "first_pass_errors": 0},
    "el_salvador_50000_english.jsonl": {"runtime_min": 44.0, "pods": 2, "first_pass_errors": 0},
    "el_salvador_50000_spanish.jsonl": {"runtime_min": 38.3, "pods": 2, "first_pass_errors": 0},
    "sv_cultural_50k_spanish.jsonl": {"runtime_min": 52.9, "pods": 2, "first_pass_errors": 0},
    "sv_cultural_50k_english.jsonl": {"runtime_min": 42.2, "pods": 2, "first_pass_errors": 0},
    "sv_persona_50k_spanish.jsonl": {"runtime_min": 41.1, "pods": 2, "first_pass_errors": 0},
    "sv_persona_50k_english.jsonl": {"runtime_min": 39.2, "pods": 2, "first_pass_errors": 0},
    "brazil_short_portuguese_demographic.jsonl": {"runtime_min": 52.0, "pods": 2, "first_pass_errors": 0},
    "brazil_short_english_demographic.jsonl": {"runtime_min": 44.2, "pods": 2, "first_pass_errors": 0},
    "brazil_short_portuguese_cultural.jsonl": {"runtime_min": 62.5, "pods": 2, "first_pass_errors": 1},
    "brazil_short_english_cultural.jsonl": {"runtime_min": 133.0, "pods": 2, "first_pass_errors": 605},
    "brazil_translation_phase1_missing_v2.jsonl": {"runtime_min": 43.7, "pods": 1, "first_pass_errors": 0},
}


def read_checkpoint(path, candidates, cache=True):
    """One row per person from a personasurvey checkpoint.

    Retried requests appear several times in a checkpoint; the successful record
    is kept when there is one. A response is `valid` when it is `ok` and the
    candidate probabilities sum to 100.
    """
    path = Path(path)
    cache_path = CACHE_DIR / f"{path.stem}.parquet"
    if cache and cache_path.exists() and cache_path.stat().st_mtime >= path.stat().st_mtime:
        cached = pd.read_parquet(cache_path)
        if all(c in cached.columns for c in candidates):
            return cached

    latest = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
                key = str(record["_id"])
            except (json.JSONDecodeError, KeyError):
                continue
            previous = latest.get(key)
            if previous is None or record.get("status") == "ok" or previous.get("status") != "ok":
                latest[key] = record

    rows = []
    for key, record in latest.items():
        response = record.get("response") or {}
        reason = next((response[f] for f in REASON_FIELDS if f in response), None)
        rows.append({
            "_id": key, "status": record.get("status"), "endpoint": record.get("endpoint"),
            "prompt_name": record.get("prompt_name"),
            **{c: response.get(c) for c in candidates},
            "reason": reason, "in_tok": record.get("in_tok"), "out_tok": record.get("out_tok"),
            "latency_s": record.get("latency_s"), "attempts": record.get("attempts"),
        })
    frame = pd.DataFrame(rows)
    frame[candidates] = frame[candidates].apply(pd.to_numeric, errors="coerce")
    frame["valid"] = frame["status"].eq("ok") & frame[candidates].sum(axis=1, min_count=len(candidates)).sub(100).abs().le(0.01)
    frame["winner"] = top_choice(frame, candidates).where(frame["valid"])
    if cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(cache_path, index=False)
    return frame


def top_choice(frame, candidates, suffix="", tie_label="Tie"):
    """Candidate with the highest probability per row; exact ties are not assigned."""
    values = frame[[f"{c}{suffix}" for c in candidates]].to_numpy(dtype=float)
    values = np.where(np.isnan(values), -np.inf, values)
    tied = np.isclose(values, values.max(axis=1, keepdims=True)).sum(axis=1) > 1
    choices = np.asarray(candidates, dtype=object)[values.argmax(axis=1)]
    choices[tied] = tie_label
    return pd.Series(choices, index=frame.index)


def checkpoint_usage(path):
    """Request-level usage of a checkpoint, retries included (they were paid for)."""
    path = Path(path)
    cache_path = CACHE_DIR / f"{path.stem}.usage.json"
    if cache_path.exists() and cache_path.stat().st_mtime >= path.stat().st_mtime:
        return json.loads(cache_path.read_text())
    lines, ids, ok = 0, set(), set()
    in_tok = out_tok = latency = 0.0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            lines += 1
            ids.add(record.get("_id"))
            if record.get("status") == "ok":
                ok.add(record.get("_id"))
            in_tok += record.get("in_tok") or 0
            out_tok += record.get("out_tok") or 0
            latency += record.get("latency_s") or 0
    usage = {"checkpoint": path.name, "requests": lines, "people": len(ids), "ok_people": len(ok),
             "input_tokens": int(in_tok), "output_tokens": int(out_tok),
             "mean_latency_s": latency / max(lines, 1)}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(usage))
    return usage


def vote_shares(winners, candidates):
    """Share of people (%) whose top choice is each candidate; ties counted separately."""
    counts = winners.value_counts().reindex([*candidates, "Tie"], fill_value=0)
    return (counts / counts.sum() * 100).rename("share_pct")


# ---------------------------------------------------------------------------
# Generic figures
# ---------------------------------------------------------------------------

def _rgba(hex_color, alpha):
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def movement_sankey(before, after, candidates, colors, left_label, right_label, measure="count", title=None):
    """Individual vote movement between two prompts for the same people.

    `before` and `after` are winner Series indexed by person. The middle column
    shows every path (stayed / A -> B), sized by the number of people.
    """
    categories = [*candidates, "Tie"] if ("Tie" in set(before) or "Tie" in set(after)) else list(candidates)
    colors = {**colors, "Tie": "#9CA3AF"}
    movement = pd.concat({"before": before, "after": after}, axis=1).dropna()
    total = len(movement)
    flow = (movement.groupby(["before", "after"]).size()
            .reindex(pd.MultiIndex.from_product([categories, categories], names=["before", "after"]), fill_value=0)
            .rename("people").reset_index())
    flow = flow.loc[flow.people.gt(0)].reset_index(drop=True)
    moved = int(movement.before.ne(movement.after).sum())
    counts_before = movement.before.value_counts().reindex(categories, fill_value=0)
    counts_after = movement.after.value_counts().reindex(categories, fill_value=0)

    def fmt(n):
        return f"{n:,} ({n / total * 100:.2f}%)" if measure == "count" else f"{n / total * 100:.2f}% ({n:,})"

    paths = [f"Stayed: {r.before}" if r.before == r.after else f"{r.before} -> {r.after}" for r in flow.itertuples()]
    labels = ([f"<b>{left_label}</b><br>{c}<br>{fmt(counts_before[c])}" for c in categories]
              + [f"{p}<br>{fmt(r.people)}" for p, r in zip(paths, flow.itertuples())]
              + [f"<b>{right_label}</b><br>{c}<br>{fmt(counts_after[c])}" for c in categories])
    n_cat, n_path = len(categories), len(flow)
    left = {c: i for i, c in enumerate(categories)}
    right = {c: n_cat + n_path + i for i, c in enumerate(categories)}
    mids = list(range(n_cat, n_cat + n_path))
    values = flow.people.tolist() if measure == "count" else (flow.people / total * 100).tolist()
    hover = [f"{r.before} -> {r.after}<br><b>{fmt(r.people)}</b>" for r in flow.itertuples()]
    fig = go.Figure(go.Sankey(
        valueformat=",.0f" if measure == "count" else ".2f",
        valuesuffix=" people" if measure == "count" else "%",
        node=dict(label=labels, pad=18, thickness=22, line=dict(color="#263A5F", width=.5),
                  color=[colors[c] for c in categories] + [colors[r.after] for r in flow.itertuples()] + [colors[c] for c in categories]),
        link=dict(source=[left[r.before] for r in flow.itertuples()] + mids,
                  target=mids + [right[r.after] for r in flow.itertuples()],
                  value=values + values,
                  color=[_rgba(colors[r.before], .55) for r in flow.itertuples()] + [_rgba(colors[r.after], .55) for r in flow.itertuples()],
                  customdata=hover + hover, hovertemplate="%{customdata}<extra></extra>"),
    ))
    title = title or f"VOTE MOVEMENT: {left_label.upper()} -> {right_label.upper()}"
    fig.update_layout(
        title=dict(text=f"<b>{title}</b><br><sup>Moved: {moved:,} ({moved / total * 100:.2f}%) | "
                        f"Stayed: {total - moved:,} ({(total - moved) / total * 100:.2f}%) | People: {total:,}</sup>",
                   x=.5, xanchor="center"),
        height=620, width=1100, margin=dict(l=15, r=15, t=95, b=20), font=dict(size=12))
    return fig


def transition_table(before, after, candidates):
    categories = [*candidates, "Tie"]
    return (pd.crosstab(before, after, rownames=["before"], colnames=["after"])
            .reindex(index=categories, columns=categories, fill_value=0)
            .loc[lambda t: t.sum(axis=1).gt(0), lambda t: t.sum(axis=0).gt(0)])


def reason_topics(reasons, topics):
    """Which election topics a model explanation mentions (simple keyword match, any language)."""
    text = reasons.fillna("").str.lower()
    return pd.DataFrame({topic: text.str.contains("|".join(words), regex=True) for topic, words in topics.items()})


USA_TOPICS = {
    "economy": ["econom", "inflation", "afford", "cost of living", "price", "job", "wage", "tax", "mortgage", "income"],
    "immigration": ["immigra", "border", "deport", "migrant"],
    "abortion": ["abortion", "reproductive", "pro-life", "pro-choice"],
}
SV_TOPICS = {
    "economy": ["econom", "bitcoin", "empleo", "employment", "job", "agric", "precio", "price", "iva", "vat", "hambre"],
    "security": ["segur", "secur", "pandill", "gang", "crime", "crimen", "delinc", "cecot", "excepci", "exception", "violen"],
    "governance": ["corrup", "gobernanza", "governance", "transparen", "separaci", "separation", "checks", "contrapeso", "instituc", "institution"],
}
BR_TOPICS = {
    "economy": ["econom", "salár", "salar", "wage", "tax", "impost", "inflaç", "inflation", "emprego", "job", "juros", "interest", "renda", "income"],
    "security": ["seguran", "secur", "crime", "crim", "violên", "violen", "polic", "facç", "faction", "prison", "pris"],
    "corruption": ["corrup", "transparên", "transparen", "govern", "stf", "supreme", "emendas", "amendment"],
}


def group_vote_table(frame, by, candidates, winner_col="winner"):
    """Vote share (%) of each candidate within each group, plus group size."""
    shares = pd.crosstab(frame[by], frame[winner_col], normalize="index").reindex(columns=candidates, fill_value=0) * 100
    shares.insert(0, "people", frame[by].value_counts())
    return shares


# ---------------------------------------------------------------------------
# USA 2024
# ---------------------------------------------------------------------------

USA_CANDIDATES = ["Donald Trump", "Kamala Harris"]
USA_COLORS = {"Donald Trump": "#C9252D", "Kamala Harris": "#246BCE", "Trump": "#C9252D", "Harris": "#246BCE"}
USA_SHORT = {"Donald Trump": "Trump", "Kamala Harris": "Harris"}
USA_DEMOGRAPHIC_COLS = ["state", "city", "ethnic_background", "sex", "age", "marital_status", "education_level", "occupation"]
USA_CHECKPOINTS = {
    "demo": ("Demographic", RESULTS_DIR / "25_AGUST_DEMO.jsonl"),
    "bg": ("Cultural Background", RESULTS_DIR / "25_AGUST_BACKGROUND.jsonl"),
    "pers": ("Persona", RESULTS_DIR / "25_AGUST_PERSONA.jsonl"),
    "career": ("Career + Big Five", RESULTS_DIR / "25_AGUST_CAREER_OCEAN.jsonl"),
}
DEMOCRATIC_STATES = ["CA", "CO", "CT", "DE", "HI", "IL", "MA", "MD", "ME", "MN", "NJ", "NM", "NY", "OR", "RI", "VT", "WA", "DC"]
REPUBLICAN_STATES = ["AL", "AR", "FL", "ID", "IN", "IA", "KS", "KY", "LA", "MO", "MS", "MT", "ND", "NE", "OH", "OK", "SC", "SD", "TN", "TX", "UT", "WV", "WY"]
SWING_STATES = ["AZ", "GA", "MI", "NC", "NV", "PA", "VA", "WI"]

EV_2024 = {"AL":9,"AK":3,"AZ":11,"AR":6,"CA":54,"CO":10,"CT":7,"DE":3,"DC":3,"FL":30,"GA":16,"HI":4,"ID":4,"IL":19,"IN":11,"IA":6,"KS":6,"KY":8,"LA":8,"ME":4,"MD":10,"MA":11,"MI":15,"MN":10,"MS":6,"MO":10,"MT":4,"NE":5,"NV":6,"NH":4,"NJ":14,"NM":5,"NY":28,"NC":16,"ND":3,"OH":17,"OK":7,"OR":8,"PA":19,"RI":4,"SC":9,"SD":3,"TN":11,"TX":40,"UT":6,"VT":3,"VA":13,"WA":12,"WV":4,"WI":10,"WY":3}
STATE_NAME = {"AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "DC": "District of Columbia", "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming"}
POPULATION_2025 = {"CA":39355309,"TX":31709821,"FL":23462518,"NY":20002427,"PA":13059432,"IL":12719141,"OH":11900510,"GA":11302748,"NC":11197968,"MI":10127884,"NJ":9548215,"VA":8880107,"WA":8001020,"AZ":7623818,"TN":7315076,"MA":7154084,"IN":6973333,"MO":6270541,"MD":6265347,"CO":6012561,"WI":5972787,"MN":5830405,"SC":5570274,"AL":5193088,"LA":4618189,"KY":4606864,"OR":4273586,"OK":4123288,"CT":3688496,"UT":3538904,"NV":3282188,"IA":3238387,"AR":3114791,"KS":2977220,"MS":2954160,"NM":2125498,"ID":2029733,"NE":2018006,"WV":1766147,"HI":1432820,"NH":1415342,"ME":1414874,"MT":1144694,"RI":1114521,"DE":1059952,"SD":935094,"ND":799358,"AK":737270,"DC":693645,"VT":644663,"WY":588753}
ELECTION_2024 = {"AL":("Trump",64.6,34.1),"AK":("Trump",54.5,40.8),"AZ":("Trump",52.2,46.7),"AR":("Trump",64.2,33.6),"CA":("Harris",38.3,58.5),"CO":("Harris",43.2,54.2),"CT":("Harris",42.4,56.4),"DE":("Harris",41.9,56.6),"DC":("Harris",6.6,90.3),"FL":("Trump",56.1,43.0),"GA":("Trump",50.7,48.5),"HI":("Harris",37.5,60.6),"ID":("Trump",66.9,30.4),"IL":("Harris",43.8,54.4),"IN":("Trump",58.6,39.7),"IA":("Trump",56.0,42.7),"KS":("Trump",57.2,41.1),"KY":("Trump",64.5,33.9),"LA":("Trump",60.2,38.2),"ME":("Harris",45.3,52.4),"MD":("Harris",34.1,63.7),"MA":("Harris",36.5,61.2),"MI":("Trump",49.7,48.3),"MN":("Harris",46.8,51.1),"MS":("Trump",60.9,37.9),"MO":("Trump",58.5,40.0),"MT":("Trump",58.4,38.5),"NE":("Trump",59.6,39.1),"NV":("Trump",50.6,47.5),"NH":("Harris",47.9,50.7),"NJ":("Harris",46.1,52.0),"NM":("Harris",45.9,51.9),"NY":("Harris",44.2,55.9),"NC":("Trump",51.0,47.7),"ND":("Trump",67.5,30.8),"OH":("Trump",55.0,43.9),"OK":("Trump",66.2,31.9),"OR":("Harris",41.1,56.0),"PA":("Trump",50.4,48.7),"RI":("Harris",41.6,55.5),"SC":("Trump",58.2,40.4),"SD":("Trump",63.4,34.2),"TN":("Trump",64.2,34.5),"TX":("Trump",56.2,42.4),"UT":("Trump",59.4,37.8),"VT":("Harris",32.4,63.8),"VA":("Harris",46.6,52.1),"WA":("Harris",39.2,58.0),"WV":("Trump",70.0,28.1),"WI":("Trump",49.6,48.7),"WY":("Trump",72.3,25.6)}
NATIONAL_2024 = {"Trump": {"popular_votes": 77303568, "popular_vote_pct": 49.81, "electoral_votes": 312},
                 "Harris": {"popular_votes": 75019230, "popular_vote_pct": 48.34, "electoral_votes": 226},
                 "Others": {"popular_votes": 2878359, "popular_vote_pct": 1.85, "electoral_votes": 0}}
US_SEX_REFERENCE = {"Female": 0.505, "Male": 0.495}
US_AGE_18PLUS = {"18-24":0.1175,"25-34":0.1740,"35-44":0.1706,"45-54":0.1527,"55-64":0.1560,"65-74":0.1328,"75-84":0.0723,"85+":0.0241}
AGE_ORDER = ["18-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75-84", "85+"]
AGE_BINS = [17, 24, 34, 44, 54, 64, 74, 84, 200]
STATE_CENTERS = {
    "AL": (32.806671, -86.791130), "AK": (61.370716, -152.404419), "AZ": (33.729759, -111.431221), "AR": (34.969704, -92.373123), "CA": (36.116203, -119.681564), "CO": (39.059811, -105.311104), "CT": (41.597782, -72.755371), "DE": (39.318523, -75.507141), "DC": (38.9072, -77.0369), "FL": (27.766279, -81.686783), "GA": (33.040619, -83.643074), "HI": (21.094318, -157.498337), "ID": (44.349426, -114.258371), "IL": (40.349457, -88.986137), "IN": (39.849426, -86.258278), "IA": (42.011539, -93.210526), "KS": (38.526600, -96.726486), "KY": (37.668140, -84.670067), "LA": (31.169546, -91.831875), "ME": (44.693947, -69.381927), "MD": (39.063946, -76.802101), "MA": (42.230171, -71.530106), "MI": (43.326618, -84.536095), "MN": (45.694454, -93.900192), "MS": (32.741646, -89.678696), "MO": (38.456085, -92.288368), "MT": (46.921925, -110.454353), "NE": (41.125370, -98.268082), "NV": (38.313515, -117.055374), "NH": (43.452492, -71.530106), "NJ": (40.298904, -74.521011), "NM": (34.840515, -106.248482), "NY": (42.165726, -74.948051), "NC": (35.630066, -79.806419), "ND": (47.528912, -99.806419), "OH": (40.388783, -82.764915), "OK": (35.565342, -96.928917), "OR": (44.572021, -122.070938), "PA": (40.590752, -77.209755), "RI": (41.680893, -71.530106), "SC": (33.856892, -80.791130), "SD": (44.299782, -100.226300), "TN": (35.747845, -86.692345), "TX": (31.054487, -97.563461), "UT": (40.150032, -111.862434), "VT": (44.045876, -72.571564), "VA": (37.769337, -78.169968), "WA": (47.400902, -121.490494), "WV": (38.491646, -80.954453), "WI": (44.268543, -89.616508), "WY": (42.755966, -107.302490),
}
REAL_2024 = pd.DataFrame.from_dict(ELECTION_2024, orient="index", columns=["real_winner", "real_trump", "real_harris"]).rename_axis("state")


def age_group(age):
    return pd.cut(pd.to_numeric(age, errors="coerce"), bins=AGE_BINS, labels=AGE_ORDER, ordered=True)


def load_usa_people(columns=("state",)):
    """The 200,023 Kish-allocated Nemotron personas; `_source_id` is the row index."""
    people = pd.read_csv(DATA_DIR / "df_personas_us_detail_kish.csv", usecols=list(columns), low_memory=False)
    people.insert(0, "_source_id", people.index.astype(str))
    if "state" in people:
        people["state"] = people["state"].astype("string").str.upper().str.strip()
    return people


def build_legacy_prompt(row):
    """Trial 1 manual prompt, kept verbatim: the stability tests used it."""
    return f'''{PROMPTS["usa"]["base_context"]}

DEMOGRAPHIC PROFILE:
State: {row["state"]}
City: {row["city"]}
Race/Ethnicity: {row["ethnic_background"]}
Sex: {row["sex"]}
Age: {row["age"]}
Marital status: {row["marital_status"]}
Education: {row["education_level"]}
Occupation: {row["occupation"]}

{PROMPTS["usa"]["election_context"]}

Return ONLY valid JSON. The two numbers must add up to 100:
{{
  "Donald Trump": probability of voting for Donald Trump,
  "Kamala Harris": probability of voting for Kamala Harris,
  "reason": "a brief explanation of the main reason for the prediction, based on the most relevant aspects of the persona and demographic"
}}'''


def usa_prompts():
    """The four USA prompt arms as personasurvey `PersonaPrompt` objects."""
    from personasurvey import PersonaPrompt, ResponseSchema
    schema = ResponseSchema.probabilities(USA_CANDIDATES)
    common = dict(context=PROMPTS["usa"]["base_context"], demographic_cols=USA_DEMOGRAPHIC_COLS,
                  scenario=PROMPTS["usa"]["election_context"], response_schema=schema)
    return {
        "demo": PersonaPrompt(name="usa_demographic", **common),
        "bg": PersonaPrompt(name="usa_cultural", persona_detail_cols=["cultural_background"], **common),
        "pers": PersonaPrompt(name="usa_persona", persona_detail_cols=["persona"], **common),
        "career": PersonaPrompt(name="usa_career_ocean", persona_detail_cols=["career_goals_and_ambitions"], include_big_five=True, **common),
    }


def usa_model(prefix, people_states, label=None, path=None):
    """Person-level votes of one USA arm aggregated to states and the Electoral College."""
    default_label, default_path = USA_CHECKPOINTS[prefix]
    label, path = label or default_label, path or default_path
    votes = read_checkpoint(path, USA_CANDIDATES).rename(columns={"_id": "_source_id"})
    merged = people_states.merge(votes, on="_source_id", how="right", validate="one_to_one")
    people = merged.loc[merged["valid"] & merged["state"].notna()]
    by_state = people.groupby("state")[USA_CANDIDATES].mean().rename(columns={"Donald Trump": f"{prefix}_trump", "Kamala Harris": f"{prefix}_harris"})
    by_state[f"{prefix}_winner"] = np.where(by_state[f"{prefix}_trump"] > by_state[f"{prefix}_harris"], "Trump", "Harris")
    state = (REAL_2024.join(by_state, how="left")
             .join(pd.Series(EV_2024, name="electoral_votes")).join(pd.Series(POPULATION_2025, name="population")))
    assert state[f"{prefix}_winner"].notna().all(), f"{label} is missing state results."
    assert int(state["electoral_votes"].sum()) == 538
    state[f"{prefix}_correct"] = state[f"{prefix}_winner"].eq(state["real_winner"])
    state[f"{prefix}_bias"] = state[f"{prefix}_trump"] - state["real_trump"]
    weights = state["population"] / state["population"].sum()
    summary = {
        "checkpoint": Path(path).name, "people": len(merged), "valid": int(merged["valid"].sum()),
        "errors": int((~merged["valid"]).sum()),
        "trump_popular": float((state[f"{prefix}_trump"] * weights).sum()),
        "harris_popular": float((state[f"{prefix}_harris"] * weights).sum()),
        "trump_ev": int(state.loc[state[f"{prefix}_winner"].eq("Trump"), "electoral_votes"].sum()),
        "harris_ev": int(state.loc[state[f"{prefix}_winner"].eq("Harris"), "electoral_votes"].sum()),
        "input_tokens": int(votes["in_tok"].sum()), "output_tokens": int(votes["out_tok"].sum()),
        "mean_latency_s": float(votes["latency_s"].mean()),
    }
    return SimpleNamespace(merged=merged, summary=summary, state=state.reset_index(), label=label, prefix=prefix)


def print_model_summary(model):
    s = model.summary
    accuracy = model.state[f"{model.prefix}_correct"].mean() * 100
    outcome = "Electoral College tie" if s["trump_ev"] == s["harris_ev"] else ("Trump" if s["trump_ev"] > s["harris_ev"] else "Harris")
    print(f"{model.label} | checkpoint: {s['checkpoint']}")
    print(f"People: {s['valid']:,}/{s['people']:,} valid | errors: {s['errors']:,}")
    print(f"Popular vote (population weighted): Trump {s['trump_popular']:.2f}% | Harris {s['harris_popular']:.2f}%")
    print(f"Electoral College: Trump {s['trump_ev']} EV | Harris {s['harris_ev']} EV | outcome: {outcome}")
    print(f"State winner accuracy vs real 2024: {accuracy:.1f}%")


def _state_labels(frame, color="white", size=8):
    return go.Scattergeo(lon=[STATE_CENTERS[s][1] for s in frame.state], lat=[STATE_CENTERS[s][0] for s in frame.state],
                         text=frame.state, mode="text", textfont=dict(size=size, color=color), hoverinfo="skip", showlegend=False)


WINNER_SCALE = [[0, "#C9252D"], [.49, "#C9252D"], [.5, "#246BCE"], [1, "#246BCE"]]


def plot_map_vs_real(model):
    p, frame, s = model.prefix, model.state.copy(), model.summary
    accuracy = int(frame[f"{p}_correct"].sum())
    leader = "Trump" if s["trump_ev"] > s["harris_ev"] else "Harris" if s["harris_ev"] > s["trump_ev"] else "Tie"
    real_t, real_h = NATIONAL_2024["Trump"]["electoral_votes"], NATIONAL_2024["Harris"]["electoral_votes"]
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "choropleth"}, {"type": "choropleth"}]],
                        subplot_titles=(model.label.upper(), "REAL 2024"), horizontal_spacing=.03)
    for col, winner, trump, harris in [(1, f"{p}_winner", f"{p}_trump", f"{p}_harris"), (2, "real_winner", "real_trump", "real_harris")]:
        frame["code"] = frame[winner].map({"Trump": 0, "Harris": 1})
        hover = [f"<b>{r.state}</b><br>Winner: {getattr(r, winner)}<br>Trump: {getattr(r, trump):.2f}%<br>Harris: {getattr(r, harris):.2f}%<br>EV: {r.electoral_votes}" for r in frame.itertuples()]
        fig.add_trace(go.Choropleth(locations=frame.state, z=frame.code, locationmode="USA-states", colorscale=WINNER_SCALE, zmin=0, zmax=1,
                                    showscale=False, text=hover, hoverinfo="text", marker_line_color="white", marker_line_width=.8), row=1, col=col)
        fig.add_trace(_state_labels(frame), row=1, col=col)
        fig.update_geos(scope="usa", row=1, col=col)
    model_text = (f"<b>{model.label}</b><br>Trump {s['trump_ev']} EV | Harris {s['harris_ev']} EV<br>"
                  f"<b>Leader: {leader}{' by ' + str(abs(s['trump_ev'] - s['harris_ev'])) + ' EV' if leader != 'Tie' else ' -- 269-269'}</b><br>"
                  f"<b>Correct: {accuracy}/51</b> | Incorrect: {51 - accuracy}/51")
    real_text = f"<b>Real 2024</b><br>Trump {real_t} EV | Harris {real_h} EV<br><b>Leader: Trump by {real_t - real_h} EV</b>"
    for x, text in [(0.24, model_text), (0.76, real_text)]:
        fig.add_annotation(text=text, x=x, y=-.14, xref="paper", yref="paper", showarrow=False, align="center",
                           font=dict(size=14, color="#263A5F"), bgcolor="rgba(255,255,255,.98)", bordercolor="#263A5F", borderwidth=1, borderpad=7)
    fig.update_layout(title=dict(text=f"<b>ELECTORAL MAP: {model.label.upper()} VS REAL 2024</b>", x=.5, xanchor="center"),
                      height=700, width=1500, margin=dict(l=20, r=20, t=80, b=160))
    fig.show()


def plot_vote_share(model):
    p, frame, s = model.prefix, model.state.copy(), model.summary
    colors = np.where(frame[f"{p}_correct"], "#2ECC71", "#F4C542")
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scatter"}, {"type": "bar"}]],
                        subplot_titles=("STATE TRUMP SHARE: MODEL VS REAL", "POPULAR VOTE: POPULATION WEIGHTED"), horizontal_spacing=.12)
    fig.add_trace(go.Scatter(x=frame.real_trump, y=frame[f"{p}_trump"], mode="markers+text", text=frame.state, textposition="top center",
                             textfont=dict(size=8), marker=dict(size=10, color=colors, line=dict(color="#263A5F", width=.8)), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", line=dict(color="#263A5F", dash="dash"), hoverinfo="skip", showlegend=False), row=1, col=1)
    fig.add_trace(go.Bar(name=model.label, x=["Trump", "Harris"], y=[s["trump_popular"], s["harris_popular"]], marker_color=["#C9252D", "#246BCE"],
                         text=[f"{s['trump_popular']:.2f}%", f"{s['harris_popular']:.2f}%"], textposition="outside"), row=1, col=2)
    fig.add_trace(go.Bar(name="Real 2024", x=["Trump", "Harris"], y=[NATIONAL_2024["Trump"]["popular_vote_pct"], NATIONAL_2024["Harris"]["popular_vote_pct"]],
                         marker_color=["rgba(201,37,45,.35)", "rgba(36,107,206,.35)"], textposition="outside"), row=1, col=2)
    fig.update_xaxes(title_text="Real Trump vote share (%)", range=[0, 100], row=1, col=1)
    fig.update_yaxes(title_text="Model Trump vote share (%)", range=[0, 100], row=1, col=1)
    fig.update_yaxes(title_text="Vote share (%)", range=[0, 100], row=1, col=2)
    fig.update_layout(barmode="group", title=dict(text=f"<b>VOTE-SHARE CALIBRATION: {model.label.upper()}</b><br><sup>Green = correct state winner | Yellow = incorrect</sup>", x=.5, xanchor="center"),
                      height=610, width=1450, margin=dict(l=50, r=35, t=100, b=70))
    fig.show()


def plot_bias(model):
    p, frame = model.prefix, model.state.copy()
    bias = f"{p}_bias"
    limit = max(float(frame[bias].abs().max()), 1)
    fig = go.Figure(go.Choropleth(locations=frame.state, z=frame[bias], locationmode="USA-states", colorscale=[[0, "#2166AC"], [.5, "#F7F7F7"], [1, "#B2182B"]],
                                  zmin=-limit, zmax=limit, marker_line_color="white", marker_line_width=.8, colorbar=dict(title="Trump bias<br>(pp)"),
                                  text=[f"<b>{r.state}</b><br>Bias: {getattr(r, bias):+.2f} pp" for r in frame.itertuples()], hoverinfo="text"))
    fig.add_trace(_state_labels(frame, color="black", size=9))
    fig.update_geos(scope="usa")
    fig.update_layout(title=dict(text=f"<b>STATE BIAS: {model.label.upper()} VS REAL 2024</b><br><sup>Red = overestimates Trump | Blue = overestimates Harris</sup>", x=.5, xanchor="center"),
                      height=650, width=1100, margin=dict(l=20, r=70, t=100, b=40))
    fig.show()


def plot_accuracy(model):
    p, frame = model.prefix, model.state.copy()
    correct = f"{p}_correct"
    frame["code"] = frame[correct].map({False: 0, True: 1})
    fig = go.Figure(go.Choropleth(locations=frame.state, z=frame.code, locationmode="USA-states", colorscale=[[0, "#F4C542"], [.49, "#F4C542"], [.5, "#2ECC71"], [1, "#2ECC71"]],
                                  zmin=0, zmax=1, showscale=False, marker_line_color="white", marker_line_width=.8,
                                  text=[f"<b>{r.state}</b><br>Model: {getattr(r, f'{p}_winner')}<br>Real: {r.real_winner}<br>Correct: {getattr(r, correct)}" for r in frame.itertuples()],
                                  hoverinfo="text"))
    fig.add_trace(_state_labels(frame, color="black", size=9))
    fig.update_geos(scope="usa")
    fig.add_annotation(text=f"<b>Correct: {int(frame[correct].sum())}/51</b> &nbsp; <b>Incorrect: {int((~frame[correct]).sum())}/51</b>", x=.5, y=-.12, xref="paper", yref="paper",
                       showarrow=False, font=dict(size=15, color="#263A5F"), bgcolor="white", bordercolor="#263A5F", borderwidth=1, borderpad=7)
    fig.update_layout(title=dict(text=f"<b>STATE-WINNER ACCURACY: {model.label.upper()}</b><br><sup>Green = correct | Yellow = incorrect | {frame[correct].mean() * 100:.1f}% correct</sup>", x=.5, xanchor="center"),
                      height=700, width=1100, margin=dict(l=20, r=30, t=100, b=120))
    fig.show()


def usa_comparison(models):
    """Final table of all prompt arms against the real 2024 result, plus the joined state frame."""
    rows = []
    frame = REAL_2024.join(pd.Series(EV_2024, name="electoral_votes")).join(pd.Series(POPULATION_2025, name="population"))
    for model in models:
        p, state, s = model.prefix, model.state.set_index("state"), model.summary
        frame = frame.join(state[[f"{p}_trump", f"{p}_harris", f"{p}_winner", f"{p}_correct"]])
        rows.append({"prompt": model.label, "valid people": s["valid"], "errors": s["errors"],
                     "Trump popular %": s["trump_popular"], "Harris popular %": s["harris_popular"],
                     "Trump EV": s["trump_ev"], "Harris EV": s["harris_ev"], "state accuracy %": state[f"{p}_correct"].mean() * 100,
                     "input tokens (M)": s["input_tokens"] / 1e6, "output tokens (M)": s["output_tokens"] / 1e6,
                     "mean latency (s)": s["mean_latency_s"]})
    rows.append({"prompt": "Real 2024", "Trump popular %": NATIONAL_2024["Trump"]["popular_vote_pct"], "Harris popular %": NATIONAL_2024["Harris"]["popular_vote_pct"],
                 "Trump EV": NATIONAL_2024["Trump"]["electoral_votes"], "Harris EV": NATIONAL_2024["Harris"]["electoral_votes"], "state accuracy %": 100.0})
    return pd.DataFrame(rows), frame


def plot_prompt_maps(comparison_frame, panels):
    """Side-by-side winner maps for several prompt arms; `panels` is [(prefix, label), ...], 'real' for 2024."""
    fig = make_subplots(rows=1, cols=len(panels), specs=[[{"type": "choropleth"}] * len(panels)], subplot_titles=[label for _, label in panels], horizontal_spacing=.02)
    frame = comparison_frame.reset_index()
    for col, (p, label) in enumerate(panels, 1):
        winner, trump, harris = ("real_winner", "real_trump", "real_harris") if p == "real" else (f"{p}_winner", f"{p}_trump", f"{p}_harris")
        frame["code"] = frame[winner].map({"Trump": 0, "Harris": 1})
        fig.add_trace(go.Choropleth(locations=frame.state, z=frame.code, locationmode="USA-states", colorscale=WINNER_SCALE, zmin=0, zmax=1, showscale=False,
                                    marker_line_color="white", marker_line_width=.8, hoverinfo="text",
                                    text=[f"<b>{r.state}</b><br>Winner: {getattr(r, winner)}<br>Trump: {getattr(r, trump):.2f}%<br>Harris: {getattr(r, harris):.2f}%<br>EV: {r.electoral_votes}" for r in frame.itertuples()]),
                      row=1, col=col)
        fig.add_trace(_state_labels(frame), row=1, col=col)
        fig.update_geos(scope="usa", row=1, col=col)
        t_ev = int(frame.loc[frame[winner].eq("Trump"), "electoral_votes"].sum())
        h_ev = int(frame.loc[frame[winner].eq("Harris"), "electoral_votes"].sum())
        detail = "" if p == "real" else f"<b>Correct: {int(frame[f'{p}_correct'].sum())}/51</b>"
        fig.add_annotation(text=f"<b>{label}</b><br>Trump {t_ev} EV | Harris {h_ev} EV<br>{detail}", x=(col - .5) / len(panels), y=-.12, xref="paper", yref="paper",
                           showarrow=False, align="center", font=dict(size=13, color="#263A5F"), bgcolor="rgba(255,255,255,.98)", bordercolor="#263A5F", borderwidth=1, borderpad=6)
    fig.update_layout(title=dict(text="<b>ELECTORAL MAP COMPARISON</b>", x=.5, xanchor="center"), height=640, width=560 * len(panels), margin=dict(l=20, r=20, t=90, b=140))
    fig.show()


def build_ev_sankey(comparison_frame, baseline_prefix, target_prefix, baseline_label, target_label):
    """Electoral votes: baseline winner -> path (stayed / flipped) -> target winner."""
    ev_states = comparison_frame.reset_index()[["state", "electoral_votes", f"{baseline_prefix}_winner", f"{target_prefix}_winner",
                                                f"{baseline_prefix}_trump", f"{target_prefix}_trump"]].copy()
    ev_states.columns = ["state", "ev", "baseline", "target", "baseline_trump", "target_trump"]
    assert len(ev_states) == 51 and int(ev_states.ev.sum()) == 538
    ev_states["trump_change_pp"] = ev_states.target_trump - ev_states.baseline_trump
    ev_states["path"] = np.select(
        [ev_states.baseline.eq("Trump") & ev_states.target.eq("Trump"), ev_states.baseline.eq("Trump") & ev_states.target.eq("Harris"),
         ev_states.baseline.eq("Harris") & ev_states.target.eq("Trump")],
        ["Stayed Trump", "Trump -> Harris flip", "Harris -> Trump flip"], default="Stayed Harris")
    path_ev = ev_states.groupby("path")["ev"].sum()
    order = [p for p in ["Stayed Trump", "Trump -> Harris flip", "Harris -> Trump flip", "Stayed Harris"] if path_ev.get(p, 0) > 0]
    destination = {"Stayed Trump": "Trump", "Trump -> Harris flip": "Harris", "Harris -> Trump flip": "Trump", "Stayed Harris": "Harris"}
    origin = {"Stayed Trump": "Trump", "Trump -> Harris flip": "Trump", "Harris -> Trump flip": "Harris", "Stayed Harris": "Harris"}
    base_c = [c for c in ["Trump", "Harris"] if c in set(ev_states.baseline)]
    targ_c = [c for c in ["Trump", "Harris"] if c in set(ev_states.target)]
    nodes = [f"L:{c}" for c in base_c] + order + [f"R:{c}" for c in targ_c]
    ids = {name: i for i, name in enumerate(nodes)}
    labels = ([f"<b>{baseline_label}</b><br>{c} {int(ev_states.loc[ev_states.baseline.eq(c), 'ev'].sum())} EV" for c in base_c]
              + [f"{p}<br>{int(path_ev[p])} EV" for p in order]
              + [f"<b>{target_label}</b><br>{c} {int(ev_states.loc[ev_states.target.eq(c), 'ev'].sum())} EV" for c in targ_c])
    values = [int(path_ev[p]) for p in order]
    fig = go.Figure(go.Sankey(
        valueformat=",.0f", valuesuffix=" EV",
        node=dict(label=labels, pad=15, thickness=20, line=dict(color="#263A5F", width=.5),
                  color=[USA_COLORS[c] for c in base_c] + [USA_COLORS[destination[p]] for p in order] + [USA_COLORS[c] for c in targ_c]),
        link=dict(source=[ids[f"L:{origin[p]}"] for p in order] + [ids[p] for p in order],
                  target=[ids[p] for p in order] + [ids[f"R:{destination[p]}"] for p in order],
                  value=values + values,
                  color=[_rgba(USA_COLORS[origin[p]], .6) for p in order] + [_rgba(USA_COLORS[destination[p]], .6) for p in order]),
    ))
    flipped = ev_states.loc[ev_states.baseline.ne(ev_states.target)].copy()
    fig.update_layout(title=dict(text=f"<b>ELECTORAL VOTE MOVEMENT: {baseline_label.upper()} -> {target_label.upper()}</b>"
                                      f"<br><sup>{int(flipped.ev.sum())} of 538 EV changed winner ({len(flipped)} state{'s' if len(flipped) != 1 else ''})</sup>",
                                 x=.5, xanchor="center"), font=dict(size=12), height=520, width=820, margin=dict(l=10, r=10, t=90, b=20))
    return fig, ev_states, flipped


def flip_table(ev_flips, baseline_label, target_label):
    table = ev_flips.sort_values("ev", ascending=False)[["state", "ev", "baseline", "target", "baseline_trump", "target_trump", "trump_change_pp"]].copy()
    table.columns = ["State", "EV", f"{baseline_label} winner", f"{target_label} winner", f"{baseline_label} Trump %", f"{target_label} Trump %", "Trump change (pp)"]
    return table.round(2).reset_index(drop=True)


def usa_people_winners(model):
    frame = model.merged.loc[model.merged["valid"]].set_index("_source_id")
    return frame["winner"].replace(USA_SHORT)


def show_usa_movement(baseline, target, comparison_frame):
    """People Sankey, Electoral-vote Sankey and the table of state flips for one comparison."""
    from IPython.display import display
    movement_sankey(usa_people_winners(baseline), usa_people_winners(target), ["Trump", "Harris"], USA_COLORS,
                    baseline.label, target.label, title=f"INDIVIDUAL VOTE MOVEMENT: {baseline.label.upper()} -> {target.label.upper()}").show()
    fig_ev, ev_states, flips = build_ev_sankey(comparison_frame, baseline.prefix, target.prefix, baseline.label, target.label)
    fig_ev.show()
    display(flip_table(flips, baseline.label, target.label))
    return ev_states, flips


# ---------------------------------------------------------------------------
# El Salvador 2024
# ---------------------------------------------------------------------------

SV_CANDIDATES = ["Nayib Bukele", "Manuel Flores", "Joel Sánchez"]
SV_COLORS = {"Nayib Bukele": "#0F766E", "Manuel Flores": "#C2410C", "Joel Sánchez": "#4F46E5"}
SV_DEMOGRAPHIC_COLS = ["department", "sex", "age", "marital_status", "education_level", "age_group"]
SV_FIELD_LABELS = {
    "en": {"department": "Department", "sex": "Sex", "age": "Age", "marital_status": "Marital status", "education_level": "Education level", "age_group": "Age group"},
    "es": {"department": "Departamento", "sex": "Sexo", "age": "Edad", "marital_status": "Estado civil", "education_level": "Nivel educativo", "age_group": "Grupo de edad"},
}
SV_VALUE_MAPS = {
    "en": {"sex": {"female": "female", "male": "male"},
           "marital_status": {"casado": "married", "divorciado": "divorced", "separado": "separated", "soltero": "single", "union_libre": "cohabiting", "viudo": "widowed"},
           "education_level": {"bachillerato": "high school diploma", "ninguno": "none", "posgrado": "postgraduate", "primaria": "primary school",
                               "secundaria": "secondary school", "tecnico": "technical education", "universitario": "university education"}},
    "es": {"sex": {"female": "femenino", "male": "masculino"},
           "marital_status": {"casado": "casado", "divorciado": "divorciado", "separado": "separado", "soltero": "soltero", "union_libre": "unión libre", "viudo": "viudo"},
           "education_level": {"bachillerato": "bachillerato", "ninguno": "ninguno", "posgrado": "posgrado", "primaria": "primaria",
                               "secundaria": "secundaria", "tecnico": "técnico", "universitario": "universitario"}},
}
SV_CHECKPOINTS = {
    ("demographic", "en"): RESULTS_DIR / "el_salvador_50000_english.jsonl",
    ("demographic", "es"): RESULTS_DIR / "el_salvador_50000_spanish.jsonl",
    ("cultural", "en"): RESULTS_DIR / "sv_cultural_50k_english.jsonl",
    ("cultural", "es"): RESULTS_DIR / "sv_cultural_50k_spanish.jsonl",
    ("persona", "en"): RESULTS_DIR / "sv_persona_50k_english.jsonl",
    ("persona", "es"): RESULTS_DIR / "sv_persona_50k_spanish.jsonl",
}
SV_PILOT_CHECKPOINTS = {"en": RESULTS_DIR / "el_salvador_500_english.jsonl", "es": RESULTS_DIR / "el_salvador_500_spanish.jsonl"}
SV_PILOT_SIZE, SV_SEED = 500, 56060

# VII Censo de Población y VI de Vivienda 2024 (BCR) and TSE Anexo 9-C, presidential results 2024.
SV_POPULATION_BY_DEPARTMENT = {
    "San Salvador": 1_563_371, "La Libertad": 765_879, "Santa Ana": 552_938, "Sonsonate": 470_455, "San Miguel": 447_634,
    "Ahuachapán": 348_880, "Usulután": 325_494, "La Paz": 318_374, "Cuscatlán": 244_901, "La Unión": 224_375,
    "Chalatenango": 185_930, "Morazán": 169_784, "San Vicente": 161_857, "Cabañas": 143_049,
}
SV_AGE_18PLUS = {"18-24": 0.1540, "25-34": 0.2300, "35-44": 0.1837, "45-54": 0.1652, "55-64": 0.1242, "65-74": 0.0812, "75-84": 0.0441, "85+": 0.0176}
SV_SEX_18PLUS = {"female": 0.540798, "male": 0.459202}
SV_ELECTION_2024_BY_DEPARTMENT = pd.DataFrame.from_dict({
    "San Salvador": [50_165, 54_496, 4_930, 7_121, 702_023, 30_956], "Santa Ana": [12_968, 11_578, 2_429, 1_695, 227_316, 3_976],
    "San Miguel": [6_811, 14_262, 2_573, 960, 159_060, 1_651], "La Libertad": [27_208, 20_725, 2_745, 2_744, 302_425, 16_197],
    "Usulután": [5_445, 14_965, 1_829, 536, 116_672, 811], "Sonsonate": [15_107, 14_920, 2_251, 1_649, 188_162, 2_546],
    "La Unión": [4_040, 4_260, 366, 234, 86_077, 400], "La Paz": [10_334, 9_582, 1_968, 961, 121_780, 1_487],
    "Chalatenango": [7_502, 13_029, 516, 551, 67_309, 886], "Cuscatlán": [10_078, 10_288, 529, 690, 99_970, 1_384],
    "Ahuachapán": [11_568, 9_355, 1_754, 911, 141_263, 1_463], "Morazán": [4_586, 10_905, 361, 289, 61_079, 339],
    "San Vicente": [5_147, 7_827, 687, 375, 56_028, 598], "Cabañas": [4_984, 5_122, 360, 364, 49_916, 556],
}, orient="index", columns=["ARENA", "FMLN", "Fuerza Solidaria", "FPS", "Nuevas Ideas", "Nuestro Tiempo"])
SV_ELECTION_2024_EXTERIOR = {"ARENA": 1_938, "FMLN": 2_853, "Fuerza Solidaria": 175, "FPS": 213, "Nuevas Ideas": 322_645, "Nuestro Tiempo": 1_826}
SV_NATIONAL_2024 = {"ARENA": 177_881, "FMLN": 204_167, "Fuerza Solidaria": 23_473, "FPS": 19_293, "Nuevas Ideas": 2_701_725, "Nuestro Tiempo": 65_076}
SV_PARTY_CANDIDATE = {"Nuevas Ideas": "Nayib Bukele", "FMLN": "Manuel Flores", "ARENA": "Joel Sánchez"}


def sv_real_shares(by_department=False):
    """Real 2024 vote share (%) of the three simulated candidates among all valid votes."""
    table = SV_ELECTION_2024_BY_DEPARTMENT if by_department else pd.DataFrame([SV_NATIONAL_2024], index=["El Salvador"])
    shares = table.div(table.sum(axis=1), axis=0) * 100
    return shares[list(SV_PARTY_CANDIDATE)].rename(columns=SV_PARTY_CANDIDATE)


def load_sv_people(columns=None):
    """The 50,000 PGM census-adjusted personas. Demographic runs use `sv_{row:05d}`, the others use `uuid`."""
    people = pd.read_parquet(DATA_DIR / "es_SV_pgm_census_50k.parquet", columns=columns)
    people.insert(0, "_source_id", [f"sv_{i:05d}" for i in range(len(people))])
    return people


def translated_demographics(row, columns, labels, value_maps, heading):
    lines = [f"{labels[c]}: {value_maps.get(c, {}).get(str(row[c]), str(row[c]))}" for c in columns]
    return heading + ":\n" + "\n".join(lines)


def build_sv_prompt(row, language):
    """Demographic-arm prompt for El Salvador in English (`en`) or Spanish (`es`)."""
    ctx = PROMPTS["el_salvador"]
    heading = "DEMOGRAPHIC PROFILE" if language == "en" else "PERFIL DEMOGRÁFICO"
    return "\n\n".join([ctx[f"base_context_{language}"],
                        translated_demographics(row, SV_DEMOGRAPHIC_COLS, SV_FIELD_LABELS[language], SV_VALUE_MAPS[language], heading),
                        ctx[f"election_context_{language}"], ctx[f"response_{language}"]])


def load_sv_run(arm, language, people=None):
    """One El Salvador arm joined to the persona demographics (keyed by `uuid`)."""
    people = people if people is not None else load_sv_people(["uuid", *SV_DEMOGRAPHIC_COLS, "area"])
    votes = read_checkpoint(SV_CHECKPOINTS[(arm, language)], SV_CANDIDATES)
    key = "_source_id" if arm == "demographic" else "uuid"
    merged = people.merge(votes.rename(columns={"_id": key}), on=key, how="inner", validate="one_to_one")
    merged["arm"], merged["language"] = arm, language
    return merged


def sv_department_maps(frames, title):
    """Winner by department for several conditions, switchable from a menu. `frames` maps label -> (frame, winner column)."""
    from salvador_department_map import build_department_maps
    return build_department_maps(frames, SV_CANDIDATES, DATA_DIR / "el_salvador_departments.geojson", title)


def build_vote_ranking(frame, choice_col, label, candidates):
    counts = frame[choice_col].value_counts()
    ranking = pd.DataFrame({"condition": label, "candidate": list(candidates), "votes": [int(counts.get(c, 0)) for c in candidates]})
    ranking["percentage"] = ranking["votes"] / len(frame) * 100
    ranking = ranking.sort_values("votes", ascending=False, kind="stable").reset_index(drop=True)
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    ranking["ties_not_assigned"] = int(counts.get("Tie", 0))
    return ranking


def bootstrap_mean_ci(values, seed, iterations=5_000):
    values = np.asarray(values, dtype=float)
    boot = np.random.default_rng(seed).choice(values, size=(iterations, len(values)), replace=True).mean(axis=1)
    return np.quantile(boot, [0.025, 0.975])


# ---------------------------------------------------------------------------
# Brazil 2026
# ---------------------------------------------------------------------------

BR_CANDIDATES = ["Luiz Inácio Lula da Silva", "Flávio Bolsonaro"]
BR_SHORT = {"Luiz Inácio Lula da Silva": "Lula", "Flávio Bolsonaro": "Flávio Bolsonaro"}
BR_COLORS = {"Luiz Inácio Lula da Silva": "#C8102E", "Flávio Bolsonaro": "#1D4E89"}
BR_REQUIRED_COLUMNS = ["sex", "marital_status", "education_level", "region", "age"]
BR_FIELD_LABELS = {
    "en": {"sex": "Sex", "marital_status": "Marital status", "education_level": "Education level", "region": "Region", "age": "Age"},
    "pt": {"sex": "Sexo", "marital_status": "Estado civil", "education_level": "Nível educacional", "region": "Região", "age": "Idade"},
}
BR_VALUE_MAPS = {
    "en": {"sex": {"Feminino": "female", "Masculino": "male"},
           "marital_status": {"Casado": "married", "Divorciado": "divorced", "Desquitado ou separado judicialmente": "legally separated", "Solteiro": "single", "Viúvo": "widowed"},
           "education_level": {"Sem instrução e fundamental incompleto": "no education and elementary incomplete",
                               "Fundamental completo e médio incompleto": "elementary complete and high school incomplete",
                               "Médio completo e superior incompleto": "high school complete and university incomplete",
                               "Superior completo": "university complete"}},
    "pt": {"sex": {"Feminino": "feminino", "Masculino": "masculino"},
           "marital_status": {"Casado": "casado", "Divorciado": "divorciado", "Desquitado ou separado judicialmente": "desquitado ou separado judicialmente", "Solteiro": "solteiro", "Viúvo": "viúvo"},
           "education_level": {"Sem instrução e fundamental incompleto": "sem instrução e fundamental incompleto",
                               "Fundamental completo e médio incompleto": "fundamental completo e médio incompleto",
                               "Médio completo e superior incompleto": "médio completo e superior incompleto",
                               "Superior completo": "superior completo"}},
}
BR_ARMS = ["demographic", "cultural", "persona"]
BR_LANGUAGE_NAME = {"pt": "portuguese", "en": "english"}
BR_CHECKPOINTS = {
    **{("short", arm, lang): DATA_DIR / f"brazil_short_{BR_LANGUAGE_NAME[lang]}_{arm}.jsonl" for arm in BR_ARMS for lang in ("pt", "en")},
    **{("full", arm, lang): RESULTS_DIR / f"brazil_full_{BR_LANGUAGE_NAME[lang]}_{arm}.jsonl" for arm in ("demographic", "cultural") for lang in ("pt", "en")},
}
BR_CANDIDATE_CHANGE = {"original": DATA_DIR / "brazil_candidate_change_pt_original_2000.jsonl",
                       "swapped": DATA_DIR / "brazil_candidate_change_pt_swapped_2000.jsonl"}
BR_CANDIDATE_CHANGE_N, BR_CANDIDATE_CHANGE_SEED = 2_000, 2026

# IBGE, Censo Demográfico 2022: population by federative unit (total 203,080,756; 51.5% women).
BR_POPULATION_2022 = {
    "São Paulo": 44_411_238, "Minas Gerais": 20_539_989, "Rio de Janeiro": 16_055_174, "Bahia": 14_141_626, "Paraná": 11_444_380,
    "Rio Grande do Sul": 10_882_965, "Pernambuco": 9_058_931, "Ceará": 8_794_957, "Pará": 8_120_131, "Santa Catarina": 7_610_361,
    "Goiás": 7_056_495, "Maranhão": 6_776_699, "Paraíba": 3_974_687, "Amazonas": 3_941_613, "Espírito Santo": 3_833_712,
    "Mato Grosso": 3_658_649, "Rio Grande do Norte": 3_302_729, "Piauí": 3_271_199, "Alagoas": 3_127_683, "Distrito Federal": 2_817_381,
    "Mato Grosso do Sul": 2_757_013, "Sergipe": 2_210_004, "Rondônia": 1_581_196, "Tocantins": 1_511_460, "Acre": 830_018,
    "Amapá": 733_759, "Roraima": 636_707,
}
BR_SEX_2022 = {"Feminino": 0.515, "Masculino": 0.485}
BR_ELECTION_YEAR = 2026  # Lula vs Flávio Bolsonaro; no official result exists yet, so there is no real-vote benchmark.


def load_brazil_people(columns=None):
    """The 50,000 pt_BR Nemotron personas with their English translations (`*_en`)."""
    usecols = None if columns is None else list(dict.fromkeys(["_source_id", *columns]))
    return pd.read_csv(DATA_DIR / "df_english_brazil50kfull.csv", usecols=usecols)


def add_language_columns(df):
    """`<column>_pt` / `<column>_en` display columns: PersonaPrompt prints values as-is."""
    df = df.copy()
    for column in ["sex", "marital_status", "education_level"]:
        for lang in ("pt", "en"):
            df[f"{column}_{lang}"] = df[column].map(BR_VALUE_MAPS[lang][column])
    for column in ["age", "region"]:
        df[f"{column}_pt"] = df[column]
        df[f"{column}_en"] = df[column]
    return df


def brazil_prompts(context="short"):
    """The three Brazil arms in Portuguese and English for the `short` or `full` proposal context."""
    from personasurvey import PersonaPrompt, ResponseSchema
    ctx = PROMPTS["brazil"]
    schemas = {"pt": ResponseSchema.probabilities(BR_CANDIDATES, reason_field="razao"),
               "en": ResponseSchema.probabilities(BR_CANDIDATES, reason_field="reason")}
    detail = {"cultural": {"pt": {"CONTEXTO CULTURAL": "cultural_background"}, "en": {"CULTURAL BACKGROUND": "cultural_background_en"}},
              "persona": {"pt": {"PERSONA": "persona"}, "en": {"PERSONA": "persona_en"}}}
    prompts = {}
    for lang in ("pt", "en"):
        demo_cols = {BR_FIELD_LABELS[lang][c]: f"{c}_{lang}" for c in BR_REQUIRED_COLUMNS}
        common = dict(context=ctx[f"base_context_{lang}"], demographic_cols=demo_cols,
                      scenario=ctx[f"election_context_{context}_{lang}"], response_schema=schemas[lang])
        prompts[("demographic", lang)] = PersonaPrompt(name=f"br_{context}_demo_{lang}", **common)
        for arm in ("cultural", "persona"):
            prompts[(arm, lang)] = PersonaPrompt(name=f"br_{context}_{arm}_{lang}", persona_detail_cols=detail[arm][lang], **common)
    return prompts


def swap_candidate_labels_original(text, candidate_a, candidate_b):
    """The swap used in the September 2026 run. It only exchanges the *full* names, so the stance
    paragraphs labelled with the short name "Lula" keep that label: in the swapped prompt both
    programmes are attributed to Lula and Flávio Bolsonaro has none. Kept to document the flaw."""
    marker = "__CANDIDATE_SWAP_MARKER__"
    assert marker not in text
    return text.replace(candidate_a, marker).replace(candidate_b, candidate_a).replace(marker, candidate_b)


def swap_candidate_labels(text):
    """Exchange Lula and Flávio Bolsonaro everywhere (full names and the short stance labels),
    keeping every proposal paragraph in its place: each programme now carries the other candidate's name."""
    import re
    lula_full, flavio = BR_CANDIDATES
    a, b, c = "\x00A\x00", "\x00B\x00", "\x00C\x00"
    text = text.replace(lula_full, a).replace(flavio, b)
    text = re.sub(r"\bLula\b", c, text)
    text = re.sub(re.escape(b) + r"(?='s stance:|:)", "Lula", text)
    text = text.replace(b, lula_full).replace(a, flavio).replace(c, flavio)
    assert "\x00" not in text
    return text


def br_share_maps(shares, title, candidate="Luiz Inácio Lula da Silva"):
    """Choropleth of a candidate's vote share (%) by federative unit; `shares` maps label -> Series indexed by state name.
    A menu switches between labels."""
    geojson = json.loads((DATA_DIR / "brazil_states.geojson").read_text(encoding="utf-8"))
    fig = go.Figure()
    for i, (label, values) in enumerate(shares.items()):
        fig.add_trace(go.Choropleth(geojson=geojson, featureidkey="properties.name", locations=values.index, z=values.values,
                                    zmin=0, zmax=100, colorscale=[[0, BR_COLORS["Flávio Bolsonaro"]], [.5, "#F7F7F7"], [1, BR_COLORS[candidate]]],
                                    colorbar=dict(title=f"{BR_SHORT[candidate]} %"), marker_line_color="white", visible=i == 0,
                                    text=[f"{s}: {v:.1f}%" for s, v in values.items()], hoverinfo="text", name=label))
    fig.update_geos(fitbounds="locations", visible=False)
    buttons = [dict(label=label, method="update", args=[{"visible": [j == i for j in range(len(shares))]}, {"title.text": f"{title} · {label}"}])
               for i, label in enumerate(shares)]
    fig.update_layout(title=f"{title} · {next(iter(shares))}", height=620, margin=dict(l=10, r=10, t=90, b=10),
                      updatemenus=[dict(buttons=buttons, x=1, y=1.12, xanchor="right")])
    return fig


def load_brazil_run(context, arm, language, people=None):
    people = people if people is not None else load_brazil_people(BR_REQUIRED_COLUMNS + ["race", "religion"])
    votes = read_checkpoint(BR_CHECKPOINTS[(context, arm, language)], BR_CANDIDATES)
    merged = people.merge(votes.rename(columns={"_id": "_source_id"}), on="_source_id", how="inner", validate="one_to_one")
    merged["context"], merged["arm"], merged["language"] = context, arm, language
    return merged
