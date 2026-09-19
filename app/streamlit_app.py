"""Grounding the Electorate — run with: py -m streamlit run app/streamlit_app.py"""
from __future__ import annotations

import base64
import difflib
import html
import json
import sys
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0,str(Path(__file__).resolve().parent))
from explorer_data import APP_DIR, DB, ROOT, COUNTRIES, RUNS, aggregates, current_context, detail, electoral_college, geometry, matched, pair_frames, query, run_frame, tu
from guide_copy import COPY
from locales import LANGUAGES, translate

TITLE='Grounding the Electorate'
AUTHOR='Jorge Pulgarin'
st.set_page_config(page_title=TITLE,page_icon='◉',layout='wide',initial_sidebar_state='collapsed')
st.markdown('<style>'+Path(__file__).with_name('explorer.css').read_text(encoding='utf-8')+'</style>',unsafe_allow_html=True)

map_component=components.declare_component('electorate_map',path=str(Path(__file__).parent/'components'/'map'))
GUIDE_RUNS=('usa_demographic_en','usa_cultural_en','usa_persona_en')
# Before/after pair shown in the guide per country. El Salvador starts at cultural because its demographic arm is unlinked.
GUIDE_PAIRS={'usa':('usa_demographic_en','usa_cultural_en'),'el_salvador':('el_salvador_cultural_es','el_salvador_persona_es'),'brazil':('brazil_demographic_pt','brazil_cultural_pt')}
FLAGS={'en':'🇺🇸','es':'🇪🇸','pt':'🇧🇷','de':'🇩🇪'}
LOCAL_LANGUAGE={'el_salvador':'es','brazil':'pt'}
ADDED_FIELD={'cultural':'cultural_background','persona':'persona','career':'career_goals_and_ambitions'}
# Verified 2026-09-18: these responses do not match the persona row their sv_{row} ID points to
# (reason age matches 2% vs 85% for uuid-keyed arms). ES and EN agree with each other, so the
# language comparison stays valid; profile, geography and cross-arm pairs do not.
UNLINKED=set()  # every El Salvador run is keyed by uuid on the corrected PGM sample (2026-09-19); nothing to hide


@st.cache_resource(show_spinner=False,max_entries=8)
def load_run(run, version): return run_frame(run)


@st.cache_data(show_spinner=False,max_entries=32)
def load_summary(run, country, metric, version): return aggregates(load_run(run,version),country,metric)


@st.cache_resource(show_spinner=False,max_entries=3)
def load_pair(a,b,version): return matched(load_run(a,version),load_run(b,version))


@st.cache_data(show_spinner=False)
def load_geo(country): return geometry(country)


@st.cache_data(show_spinner=False)
def load_detail(pid,run,version): return detail(pid,run)


@st.cache_data(show_spinner=False)
def sample_ids(runs, version, k=5, page=0):
    """Ten deterministic personas valid in every run: five whose top choice changes, five whose does not."""
    winners=pd.concat([load_run(r,version).loc[lambda f:f.valid.eq(1)].set_index('pid').winner.rename(i) for i,r in enumerate(runs)],axis=1,join='inner')
    if winners.empty: return []
    changed=winners.ne(winners[0],axis=0).any(axis=1)
    order=pd.Series(pd.util.hash_pandas_object(winners.index.to_series(),index=False).values,index=winners.index)
    def take(o):
        o=o.sort_values();start=(page*k)%max(len(o),1)
        return o.iloc[start:start+k]
    return list(pd.concat([take(order[changed]),take(order[~changed])]).sort_values().index)


@st.cache_data(show_spinner=False)
def usa_benchmark(version):
    """Trump mean probability per arm against the observed two-candidate share, plus state winners matched."""
    trump,harris=(tu.NATIONAL_2024[c]['popular_vote_pct'] for c in ('Trump','Harris'))
    rows=[]
    for run in GUIDE_RUNS:
        summary,national=load_summary(run,'usa','probability',version)
        matches=sum(1 for state,(_,t,h) in tu.ELECTION_2024.items() if state in summary.index and summary.loc[state,'valid_n']>0 and (summary.loc[state,'p0']>summary.loc[state,'p1'])==(t>h))
        rows.append(dict(arm=RUNS[run]['arm'],sim=float(national['p0']),matches=matches))
    return trump/(trump+harris)*100,rows,len(tu.ELECTION_2024)


@st.cache_data(show_spinner=False)
def language_stats(country, version):
    """Same persona, local language vs English: matched valid pairs per arm."""
    local=LOCAL_LANGUAGE[country];n=len(COUNTRIES[country]['candidates']);rows=[]
    for arm in ('demographic','cultural','persona'):
        pair=matched(load_run(f'{country}_{arm}_{local}',version),load_run(f'{country}_{arm}_en',version))
        rows.append(dict(arm=arm,n=len(pair),changed=100*pair.winner_a.ne(pair.winner_b).mean(),local=[pair[f'p{i}_a'].mean() for i in range(n)],en=[pair[f'p{i}_b'].mean() for i in range(n)]))
    return rows


@st.cache_data(show_spinner=False)
def swap_stats():
    """Corrected Brazil label swap (names and stance labels exchanged), per language, read from its checkpoints."""
    from scipy.stats import binomtest
    lula,flavio=tu.BR_CANDIDATES
    out={}
    for lang in ('pt','en'):
        paths=[tu.BR_LABEL_SWAP[(lang,c)] for c in ('original','swapped')]
        if not all(p.exists() for p in paths):continue
        o,w=(tu.read_checkpoint(p,tu.BR_CANDIDATES,cache=False) for p in paths)
        pair=o.loc[o.valid.astype(bool),['_id','winner']].merge(w.loc[w.valid.astype(bool),['_id','winner']],on='_id',suffixes=('_o','_s'))
        pair=pair.loc[pair.winner_o.isin(tu.BR_CANDIDATES)&pair.winner_s.isin(tu.BR_CANDIDATES)]
        if pair.empty:continue
        follows=pair.winner_o.map({lula:flavio,flavio:lula}).eq(pair.winner_s)
        out[lang]=dict(n=len(pair),follow=100*follows.mean(),keep=100*(1-follows.mean()),p=binomtest(int(follows.sum()),len(pair),p=.5,alternative='greater').pvalue,
                       lula_o=100*pair.winner_o.eq(lula).mean(),lula_s=100*pair.winner_s.eq(lula).mean())
    return out


if 'lang' not in st.session_state:
    st.session_state.lang=st.query_params.get('lang') if st.query_params.get('lang') in LANGUAGES.values() else 'en'
if 'guide_done' not in st.session_state and st.query_params.get('start')=='app': st.session_state.guide_done=True
if 'country' in st.session_state: st.session_state.country=st.session_state.country


def tx(key): return translate(key,st.session_state.get('lang','en'))


def g(key): return COPY[st.session_state.get('lang','en')][key]


def esc(value): return html.escape(str(value))


def short_name(name):
    """Name used in compact labels: 'Lula', not the last word 'Silva'."""
    return {'Luiz Inácio Lula da Silva':'Lula'}.get(name,name.split()[-1])

def _lum(h):
    c=[int(h.lstrip('#')[i:i+2],16)/255 for i in (0,2,4)]
    c=[x/12.92 if x<=.03928 else ((x+.055)/1.055)**2.4 for x in c]
    return .2126*c[0]+.7152*c[1]+.0722*c[2]


def _contrast(a,b):
    x,y=sorted((_lum(a),_lum(b)),reverse=True);return (x+.05)/(y+.05)


def ink(h,bg='#ffffff'):
    """Candidate color darkened just enough to read as text (WCAG AA 4.5:1); fills keep the original."""
    r,g_,b=(int(h.lstrip('#')[i:i+2],16) for i in (0,2,4));c=h
    for k in range(0,70,3):
        c='#%02x%02x%02x'%(int(r*(1-k/100)),int(g_*(1-k/100)),int(b*(1-k/100)))
        if _contrast(c,bg)>=4.5:break
    return c


def on(h): return '#ffffff' if _contrast('#ffffff',h)>=4.5 else '#1c1917'

def no_winner(totals):
    """Label for an Electoral College without 270: a true 269-269 tie, or no majority."""
    return tx('ec_tie') if totals[0] == totals[1] == 269 else tx('no_majority')


def unassigned_note(n, sep=' · '):
    return f'{sep}{n} {tx("unassigned")}' if n else ''




def robot(pid): return 'https://api.dicebear.com/10.x/bottts-neutral/svg?seed='+quote(pid)


def set_lang(code):
    st.session_state.lang=code;st.query_params['lang']=code


def language_widget(key):
    names=list(LANGUAGES)
    st.session_state[key]=next(n for n in names if LANGUAGES[n]==st.session_state.lang)
    def sync(): set_lang(LANGUAGES[st.session_state[key]])
    with st.container(key='lang_row'):st.radio('Language / Idioma',names,key=key,on_change=sync,horizontal=True,label_visibility='collapsed',format_func=lambda n:FLAGS[LANGUAGES[n]]+' '+n)


def baseline(run):
    r=RUNS[run]
    return next(k for k,v in RUNS.items() if v['country']==r['country'] and v['language']==r['language'] and v['arm']=='demographic')


def run_label(run):
    r=RUNS[run]
    several=len({v['language'] for v in RUNS.values() if v['country']==r['country']})>1
    return arm_label(r['arm'])+(' · '+r['language'].upper() if several else '')


def nice_region(region):
    return next((f['properties']['name'] for f in geo['features'] if f['properties']['id']==region),region)


def leader(values):
    top=np.flatnonzero(np.isclose(values,max(values),rtol=0,atol=1e-8))
    return int(top[0]) if len(top)==1 else -1


def response_of(data, c):
    response=data['raw'].get('response') or {}
    return [float(response[k]) for k in c['candidates']],next((response[k] for k in ('reason','razon','razao') if k in response),'')


def prob_html(values, c):
    return ''.join(f'<div class="prob-row"><span>{esc(name)}</span><b>{values[i]:.1f}%</b></div><div class="prob-track"><span style="width:{values[i]:.1f}%;background:{c["colors"][i]}"></span></div>' for i,name in enumerate(c['candidates']))


def shift(key, step, n): st.session_state[key]=(st.session_state.get(key,0)+step)%n


def person_nav(ids, key):
    i=st.session_state.get(key,0)%len(ids)
    prev,count,following=st.columns([1,.7,1])
    prev.button('← '+g('previous_person'),key=key+'_prev',on_click=shift,args=(key,-1,len(ids)),width='stretch')
    count.markdown(f'<div class="person-count">{i+1} / {len(ids)}</div>',unsafe_allow_html=True)
    following.button(g('next_person')+' →',key=key+'_next',on_click=shift,args=(key,1,len(ids)),width='stretch')


def persona_intro(pid, p, c, compact=False, status=None):
    region=p.get(c['region'],'')
    region=tu.STATE_NAME.get(region,region) if c is COUNTRIES['usa'] else region
    # El Salvador's census-adjusted demographics were redrawn and conflict with municipality/occupation
    # from the persona text; show only the fields the prompt used.
    salvador=c is COUNTRIES['el_salvador']
    place=region if salvador else ', '.join(str(x) for x in (p.get('city') or p.get('municipality'),region) if x)
    facts={g('age'):p.get('age'),g('sex'):p.get('sex',p.get('gender')),g('education'):p.get('education_level',p.get('education')),g('occupation'):None if salvador else p.get('occupation')}
    if compact:
        line=' · '.join((f'{esc(k)} ' if k==g('age') else '')+esc(str(v).replace('_',' ')) for k,v in facts.items() if v not in (None,''))
        note=f'<em class="{status[0]}">● {esc(status[1])}</em>' if status else ''
        st.markdown(f'<div class="persona-intro compact"><img src="{robot(pid)}" alt="Robot"><div><b>{esc(place)}</b><small>{esc(g("synthetic"))} · {line}</small>{note}</div></div>',unsafe_allow_html=True);return
    st.markdown(f'<div class="persona-intro"><img src="{robot(pid)}" alt="Robot"><div><small>{esc(g("synthetic"))}</small><b>{esc(place)}</b></div></div>',unsafe_allow_html=True)
    st.markdown('<div class="facts">'+''.join(f'<div><small>{esc(k)}</small><b>{esc(str(v).replace("_"," "))}</b></div>' for k,v in facts.items() if v not in (None,''))+'</div>',unsafe_allow_html=True)


def answer_html(label, values, c):
    w=leader(values)
    choice=c['candidates'][w] if w>=0 else g('tie')
    color=c['colors'][w] if w>=0 else '#5b6670'
    return f'<div class="answer"><div class="answer-label">{esc(label)}</div>{prob_html(values,c)}<div class="answer-choice">→ <b style="color:{ink(color)}">{esc(choice)}</b></div></div>'


def compare_pair(pid, run_a, run_b, c, label_a, label_b, added=True, intro_slot=None, short=False, compact=False):
    """One persona, two saved answers side by side."""
    da,db=load_detail(pid,run_a,version),load_detail(pid,run_b,version)
    if not (da and db and da['normalized']['valid'] and db['normalized']['valid']):
        st.info(g('missing'));return
    (va,ra),(vb,rb)=response_of(da,c),response_of(db,c)
    changed=leader(va)!=leader(vb)
    with (intro_slot if intro_slot is not None else st.container()):
        if {run_a,run_b}&UNLINKED:st.caption(tx('unlinked_profile'))
        else:persona_intro(pid,db['profile'],c,compact,('changed' if changed else 'same',g('changed') if changed else g('same')) if compact else None)
    if not compact:st.markdown(f'<span class="badge {"changed" if changed else "same"}">{esc(g("changed") if changed else g("same"))}</span>',unsafe_allow_html=True)
    for side,(column,label,values,reason) in enumerate(zip(st.columns(2),(label_a,label_b),(va,vb),(ra,rb))):
        with column:
            text=str(reason)
            if short and len(text)>190:text=text[:190].rsplit(' ',1)[0]+'…'
            st.markdown(answer_html(label,values,c).replace('class="answer"','class="answer b"' if side else 'class="answer"',1)+(f'<div class="reason" title="{esc(reason)}"><small>{esc(g("reason"))}</small><br>{esc(text)}</div>' if reason else ''),unsafe_allow_html=True)
    field=ADDED_FIELD.get(RUNS[run_b]['arm'])
    if added and field:
        if RUNS[run_b]['language']=='en' and RUNS[run_b]['country']!='usa' and field+'_en' in db['profile']: field+='_en'
        if db['profile'].get(field):
            with st.expander(g('details')): st.write(db['profile'][field])


# ---------- quick guide ----------

@st.cache_data(show_spinner=False)
def case_ids(run_a, run_b, version):
    """One saved persona per story: kept the top choice, moved 0 -> 1, moved 1 -> 0. Valid in both runs, no ties."""
    a,b=load_run(run_a,version),load_run(run_b,version)
    w=a.loc[a.valid.eq(1)].set_index('pid').winner.rename('a').to_frame().join(b.loc[b.valid.eq(1)].set_index('pid').winner.rename('b'),how='inner')
    w=w.loc[w.a.ge(0)&w.b.ge(0)]
    order=pd.Series(pd.util.hash_pandas_object(w.index.to_series(),index=False).values,index=w.index)
    pick=lambda mask:order[mask].idxmin() if mask.any() else None
    return [pick(w.a.eq(w.b)),pick(w.a.eq(0)&w.b.eq(1)),pick(w.a.eq(1)&w.b.eq(0))]


@st.cache_data(show_spinner=False)
def usa_board(version):
    """Every USA context: simulated Electoral College, population-weighted popular vote, state winners matched."""
    rows=[]
    for run in [r for r in RUNS if RUNS[r]['country']=='usa']:
        summary,national=load_summary(run,'usa','probability',version)
        ec=electoral_college(summary)
        matches=sum(1 for state,(_,t,h) in tu.ELECTION_2024.items() if state in summary.index and summary.loc[state,'valid_n']>0 and (summary.loc[state,'p0']>summary.loc[state,'p1'])==(t>h))
        rows.append(dict(arm=RUNS[run]['arm'],ev=ec['totals'],unassigned=ec['unassigned'],winner=ec['winner'],pop=[float(national['p0']),float(national['p1'])],matches=matches))
    return rows


@st.cache_data(show_spinner=False)
def country_board(country, version):
    """Votes per context (local language): each valid persona is one vote for its highest-probability candidate; ties count for nobody."""
    lang=LOCAL_LANGUAGE[country];n=len(COUNTRIES[country]['candidates']);rows=[]
    for arm in ('demographic','cultural','persona'):
        valid=load_run(f'{country}_{arm}_{lang}',version).loc[lambda f:f.valid.eq(1)]
        counts=[int(valid.winner.eq(i).sum()) for i in range(n)]
        rows.append(dict(arm=arm,counts=counts,total=len(valid),ties=int(valid.winner.eq(-1).sum()),values=[100*c/max(len(valid),1) for c in counts]))
    return rows


def open_guide():
    st.session_state.update(guide_done=False,guide_step=0);st.query_params.pop('start',None)


def close_guide(view=None):
    st.session_state.guide_done=True;st.query_params['start']='app'
    if view: st.session_state.main_nav=view


def step_to(i): st.session_state.guide_step=i


def set_guide_country(k): st.session_state.guide_country=k


def stacked_bar(values, colors):
    return '<div class="board-bar">'+''.join(f'<span style="width:{v}%;background:{c}"></span>' for v,c in zip(values,colors))+'</div>'


def board_row(who, sub, bar, lead, lead_sub, extra='', cls=''):
    return f'<div class="board-row {cls}"><div class="who">{esc(who)}<small>{esc(sub)}</small></div><div class="board-scale">{bar}</div><div class="lead">{lead}<small>{esc(lead_sub)}</small></div><div>{extra}</div></div>'


def arm_label(arm): return g('before') if arm=='demographic' else g(arm)


ICONS={
    'election':'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"><path d="M3 6l6-2 6 2 6-2v14l-6 2-6-2-6 2z"/><path d="M9 4v14M15 6v14"/></svg>',
    'changes':'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M4 8h15l-4-4M20 16H5l4 4"/></svg>',
    'experiments':'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"><path d="M3 4h13v10H8l-5 4z"/><path d="M16 8h5v10l-4-3h-7v-1"/></svg>',
    'about':'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"><path d="M2 5c3.5-1.3 7-1 10 1 3-2 6.5-2.3 10-1v14c-3.5-1.3-7-1-10 1-3-2-6.5-2.3-10-1z"/><path d="M12 6v14"/></svg>'}
GO_COLORS={'election':'#0f5132','changes':'#1d4e89','experiments':'#9a5b13','about':'#7a2e52'}
CHIP='<svg class="chip-ai" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"><rect x="6" y="6" width="12" height="12" rx="2"/><rect x="9.5" y="9.5" width="5" height="5" rx="1"/><path d="M9 2.5v3.5M12 2.5v3.5M15 2.5v3.5M9 18v3.5M12 18v3.5M15 18v3.5M2.5 9h3.5M2.5 12h3.5M2.5 15h3.5M18 9h3.5M18 12h3.5M18 15h3.5"/></svg>'


def thousands(n):
    text=f'{n:,}'
    return text if st.session_state.lang=='en' else text.replace(',','.')


def guide():
    step=st.session_state.get('guide_step',0);last=5
    with st.container(key='guide'):
        text,content=st.columns([1,1.35])
        with text:
            st.markdown(f'<div class="guide-step"><span class="pill">{esc(TITLE)} · {esc(g("thesis_tag"))} · {esc(AUTHOR)}</span><h2>{esc(g("titles")[step])}</h2><p>{esc(g("bodies")[step])}</p></div>',unsafe_allow_html=True)
            dots=''.join(f'<span class="{"on" if i==step else "done" if i<step else ""}"></span>' for i in range(last+1))
            st.markdown(f'<div class="guide-progress"><div class="guide-dots">{dots}</div><span class="count">{esc(g("step_of").format(n=step+1,t=last+1))}</span></div>',unsafe_allow_html=True)
            with st.container(key='guide_nav'):
                back,following=st.columns([1,1.6])
                back.button('← '+g('back'),key='guide_back',on_click=step_to,args=(max(step-1,0),),width='stretch',disabled=step==0)
                if step<last: following.button(g('next')+' →',key='guide_next',type='primary',on_click=step_to,args=(step+1,),width='stretch')
                else: following.button(g('open')+' →',key='guide_open',type='primary',on_click=close_guide,args=('election',),width='stretch')
                if step<last:st.button(g('skip')+' →',key='guide_skip',type='tertiary',on_click=close_guide,args=('election',))
        with content:
            if step==0:
                st.markdown(f'<div class="lang-label"><span>{esc(g("language"))}</span></div>',unsafe_allow_html=True)
                with st.container(key='lang_pick'):
                    for pair in (list(LANGUAGES.items())[:2],list(LANGUAGES.items())[2:]):
                        for column,(name,code) in zip(st.columns(2),pair):
                            column.button(FLAGS[code]+'  '+name,key='lang_'+code,type='primary' if code==st.session_state.lang else 'secondary',on_click=set_lang,args=(code,),width='stretch')
                personas=sum(len(load_run(r,version)) for r in ('usa_demographic_en','el_salvador_cultural_es','brazil_demographic_pt'))
                st.markdown(f'<div class="stats"><div><b>{len(COUNTRIES)}</b><small>{esc(g("stat_elections"))}</small></div><div><b>{thousands(personas)}</b><small>{esc(g("stat_personas"))}</small></div><div><b>1</b><small>{esc(g("stat_model"))}</small><code>gpt-oss-20b</code></div></div>',unsafe_allow_html=True)
            elif step==1:
                usa=COUNTRIES['usa']
                bars=''.join(f'<span style="width:{w}%;background:{c}"></span>' for w,c in ((100,usa['colors'][0]),(43,usa['colors'][1])))
                nodes=[f'<img src="{robot("usa:guide")}" alt="Robot">',CHIP,f'<div class="mini-bars">{bars}</div>']
                parts=[]
                for i,(node,name,note) in enumerate(zip(nodes,g('flow'),g('flow_notes'))):
                    if i: parts.append(f'<div class="flow-arrow" style="--d:{.6+i*.6-.3}s">→</div>')
                    parts.append(f'<div class="flow-node" style="--d:{.6+i*.6}s"><div class="tile">{node}</div><b>{esc(name)}</b><small>{esc(note)}</small></div>')
                st.markdown('<div class="flow-card"><div class="guide-flow">'+''.join(parts)+'</div></div>',unsafe_allow_html=True)
                st.markdown('<div class="pillars">'+''.join(f'<div class="pillar" style="--d:{i*.15}s"><small>{esc(t)}</small><b>{esc(n)}</b></div>' for i,(t,n) in enumerate(zip(g('pillars'),g('pillar_notes'))))+'</div>',unsafe_allow_html=True)
            elif step==2:
                with st.container(key='case_card'):
                    ctrl,who=st.columns([1.1,1],vertical_alignment='center',gap='medium')  # controls left, the voter in the free space on the right
                    with ctrl:
                        country=st.radio('Country',list(GUIDE_PAIRS),format_func=lambda k:COUNTRIES[k]['label'],horizontal=True,key='seg_voters_country',label_visibility='collapsed')
                        c=COUNTRIES[country];run_a,run_b=GUIDE_PAIRS[country]
                        ids=case_ids(run_a,run_b,version)
                        a,b=c['candidates'][:2]
                        labels=[x.format(a=short_name(a),b=short_name(b)) for x in g('cases')]
                        options=[i for i,pid in enumerate(ids) if pid]
                        case=None
                        if options:
                            with st.container(key='cases'):
                                case=st.radio(g('richer'),options,format_func=lambda i:labels[i],horizontal=True,key='seg_case_'+country,label_visibility='collapsed')
                    if case is None: st.info(g('missing'))
                    else: compare_pair(ids[case],run_a,run_b,c,arm_label(RUNS[run_a]['arm']),arm_label(RUNS[run_b]['arm']),short=True,compact=True,intro_slot=who)
            elif step==3:
                usa=COUNTRIES['usa']
                rows=usa_board(version);t,h=(tu.NATIONAL_2024[c] for c in ('Trump','Harris'));two=t['popular_vote_pct']+h['popular_vote_pct']
                body=f'<div class="board-row head"><span>{esc(g("context"))}</span><span>{esc(tx("electoral_votes"))}</span><span>{esc(g("winner"))}</span><span>{esc(g("states_matched"))}</span></div>'
                real_bar=f'<div class="board-bar"><span style="width:{t["electoral_votes"]/538*100}%;background:{usa["colors"][0]}"></span><span style="width:{h["electoral_votes"]/538*100}%;background:{usa["colors"][1]}"></span></div><span class="mark"></span><div class="board-nums"><span style="color:{ink(usa["colors"][0])}">{t["electoral_votes"]} EV · {t["popular_vote_pct"]/two*100:.1f}%</span><span style="color:{ink(usa["colors"][1])}">{h["electoral_votes"]} EV · {h["popular_vote_pct"]/two*100:.1f}%</span></div>'
                real=board_row(g('observed'),'2024',real_bar,f'<span style="color:{ink(usa["colors"][0])}">{esc(usa["candidates"][0])}</span>',f'+{t["electoral_votes"]-h["electoral_votes"]} EV','','real')
                st.markdown(f'<div class="board real-board">{real}</div>',unsafe_allow_html=True)
                for r in rows:
                    bar=f'<div class="board-bar"><span style="width:{r["ev"][0]/538*100}%;background:{usa["colors"][0]}"></span><span style="width:{r["unassigned"]/538*100}%;background:transparent"></span><span style="width:{r["ev"][1]/538*100}%;background:{usa["colors"][1]}"></span></div><span class="mark"></span><div class="board-nums"><span style="color:{ink(usa["colors"][0])}">{r["ev"][0]} EV · {r["pop"][0]:.1f}%</span><span style="color:{ink(usa["colors"][1])}">{r["ev"][1]} EV · {r["pop"][1]:.1f}%</span></div>'
                    w=r['winner']
                    lead=f'<span style="color:{ink(usa["colors"][w])}">{esc(usa["candidates"][w])}</span>' if w is not None else esc(no_winner(r['ev']))
                    body+=board_row(arm_label(r['arm']),g('simulated'),bar,lead,f'{r["ev"][w]-r["ev"][1-w]:+d} EV' if w is not None else '',f'{r["matches"]} / 51')
                st.markdown(f'<div class="board">{body}</div>',unsafe_allow_html=True)
                st.caption(g('ev_note'))
            elif step==4:
                pick=st.session_state.get('guide_country','el_salvador')
                with st.container(key='country_pick'):
                    for column,(k,flag) in zip(st.columns(2),(('el_salvador','🇸🇻'),('brazil','🇧🇷'))):
                        column.button(flag+'  '+COUNTRIES[k]['label'],key='pick_'+k,type='primary' if pick==k else 'secondary',on_click=set_guide_country,args=(k,),width='stretch')
                for country in (pick,):
                    c=COUNTRIES[country];n=len(c['candidates'])
                    title=f'<div class="board-title">{esc(c["label"])} · {LOCAL_LANGUAGE[country].upper()}</div>';body=f'<div class="board-row head"><span>{esc(g("context"))}</span><span>{esc(tx("votes_col"))}</span><span>{esc(g("winner"))}</span><span></span></div>'
                    if country=='el_salvador':
                        real=list(tu.sv_real_shares().iloc[0]);total=sum(real);real=[v/total*100 for v in real]
                        nums='<div class="board-nums">'+''.join(f'<span style="color:{ink(c["colors"][i])}">{esc(short_name(c["candidates"][i]))} {real[i]:.1f}%</span>' for i in range(n))+'</div>'
                        real_row=board_row(g('observed'),'2024',stacked_bar(real,c['colors'])+nums,f'<span style="color:{ink(c["colors"][0])}">{esc(c["candidates"][0])}</span>','','','real')
                    else:
                        real_row='';title+=f'<div class="board-sub">{esc(tx("br_dates"))}</div>'
                    st.markdown(f'<div class="board real-board">{title}{real_row}</div>',unsafe_allow_html=True)
                    for r in country_board(country,version):
                        w=leader(r['values'])
                        nums='<div class="board-nums">'+''.join(f'<span style="color:{ink(c["colors"][i])}">{esc(short_name(c["candidates"][i]))} {r["values"][i]:.1f}%</span>' for i in range(n))+'</div>'
                        lead=f'<span style="color:{ink(c["colors"][w])}">{esc(c["candidates"][w])}</span>' if w>=0 else esc(g('tie'))
                        body+=board_row(arm_label(r['arm']),g('simulated'),stacked_bar(r['values'],c['colors'])+nums,lead,f'{r["counts"][w]:,} {tx("votes")}' if w>=0 else '')
                    st.markdown(f'<div class="board">{body}</div>',unsafe_allow_html=True)
            else:
                with st.container(key='guide_go'):
                    for row in (('election','changes'),('experiments','about')):
                        for column,view in zip(st.columns(2),row):
                            with column:
                                st.markdown(f'<div class="go-card" style="--accent:{GO_COLORS[view]}"><div class="icon">{ICONS[view]}</div><b>{esc(g(view))}</b><p>{esc(g(view+"_desc"))}</p></div>',unsafe_allow_html=True)
                                st.button(g(view)+'  →',key='guide_go_'+view,on_click=close_guide,args=(view,),width='stretch')
        st.markdown(f'<div class="guide-foot"><span><b>{esc(TITLE)}</b> · {esc(g("thesis_tag"))} · {esc(AUTHOR)}</span><span>{esc(g("step_of").format(n=step+1,t=last+1))}</span></div>',unsafe_allow_html=True)


# ---------- election view ----------

def electoral_summary(summary, label):
    result=electoral_college(summary)
    totals=result['totals'];winner=result['winner']
    status=tx('sim_winner')+': '+cfg['candidates'][winner] if winner is not None else no_winner(totals)
    counts=''.join(f'<span style="color:{ink(cfg["colors"][i])}">{esc(c)} <b>{totals[i]}</b></span>' for i,c in enumerate(cfg['candidates']))
    segments=f'<span style="width:{totals[0]/538*100}%;background:{cfg["colors"][0]}"></span><span style="width:{result["unassigned"]/538*100}%;background:#d6d0c6"></span><span style="width:{totals[1]/538*100}%;background:{cfg["colors"][1]}"></span>'
    st.markdown(f'<div class="ev-summary"><div class="ev-heading"><strong>{esc(status)}</strong><span>{esc(label)}</span></div><div class="ev-counts">{counts}</div><div class="ev-scale"><div class="ev-bar">{segments}</div><span class="ev-threshold"></span></div><div class="ev-foot"><span>{esc(tx("electoral_votes"))}{esc(unassigned_note(result["unassigned"]))}</span><span>{esc(tx("majority"))}</span></div></div>',unsafe_allow_html=True)


def electoral_method():
    with st.expander(tx('ev_method')):
        st.write(tx('ev_note'))
        st.caption(tx('metric')+': '+tx(metric))
        st.markdown('[2024 electoral allocation · National Archives](https://www.archives.gov/electoral-college/allocation)')


def national_board():
    """One result strip instead of duplicate candidate and electoral KPI cards."""
    result=electoral_college(summary_b) if country=='usa' else None
    voted=b.loc[b.valid.eq(1)] if not result else None
    contenders=[]
    for i,candidate in enumerate(cfg['candidates']):
        asset=portraits.get(candidate,{})
        photo=APP_DIR/'assets'/'candidates'/asset.get('file','missing')
        img=f'<img width="52" height="62" alt="{esc(candidate)}" src="data:image/jpeg;base64,{base64.b64encode(photo.read_bytes()).decode()}">' if photo.is_file() else ''
        value=national_b.get(f'p{i}',np.nan)
        score=f'{result["totals"][i]} <small>EV</small>' if result else f'{100*voted.winner.eq(i).mean():.1f}%'
        contenders.append(f'<div class="contender">{img}<div><div class="contender-name">{esc(candidate)}</div><div class="contender-score" style="color:{ink(cfg["colors"][i])}">{score}</div><div class="contender-note">{value:.1f}% · {esc(tx(metric))}</div></div></div>' if result else f'<div class="contender">{img}<div><div class="contender-name">{esc(candidate)}</div><div class="contender-score" style="color:{ink(cfg["colors"][i])}">{score}</div><div class="contender-note">{int(voted.winner.eq(i).sum()):,} {esc(tx("votes"))}</div></div></div>')
    if result:
        winner=result['winner']
        title=cfg['candidates'][winner] if winner is not None else no_winner(result['totals'])
        totals=result['totals']
        segments=f'<span style="width:{totals[0]/538*100}%;background:{cfg["colors"][0]};color:{on(cfg["colors"][0])}">{totals[0]}</span><span style="width:{result["unassigned"]/538*100}%;background:#d6d0c6"></span><span style="width:{totals[1]/538*100}%;background:{cfg["colors"][1]};color:{on(cfg["colors"][1])}">{totals[1]}</span>'
        popular=' – '.join(f'{national_b.get(f"p{i}",np.nan):.1f}%' for i in range(len(cfg['candidates'])))
        lead=f'{esc(tx("sim_winner"))}: <b>{esc(title)}</b>' if winner is not None else f'<b>{esc(title)}</b>'
        outcome=f'<div class="outcome"><div class="ev-scale"><div class="ev-bar">{segments}</div><span class="ev-threshold"></span></div><div class="ev-foot"><span>{lead}</span>{f'<span>{esc(unassigned_note(result["unassigned"],"· "))}</span>' if result["unassigned"] else ''}<span>· {esc(tx(metric))} {popular}</span></div></div>'
        content=contenders[0]+outcome+contenders[1]
    else:
        content=''.join(contenders)
    st.markdown('<section class="scoreboard'+(' multi' if not result else '')+'" style="--candidate-count:'+str(len(contenders))+'">'+content+'</section>',unsafe_allow_html=True)


def bars(values, title='', delta=None):
    fig=go.Figure(go.Bar(y=cfg['candidates'],x=values,orientation='h',marker_color=cfg['colors'],text=[f'{x:.1f}%'+(f' ({delta[i]:+.1f} pp)' if delta is not None else '') for i,x in enumerate(values)],textposition='auto'))
    fig.update_layout(height=115+len(values)*25,margin=dict(l=0,r=20,t=22,b=10),title=dict(text=title,font_size=13),xaxis=dict(range=[0,100],visible=False),yaxis=dict(autorange='reversed'),paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font=dict(family='DM Sans',color='#254251'),showlegend=False)
    return fig


def payload(frame):
    return {str(region):dict(n=int(row.valid_n),values=[float(row[f'p{i}']) for i in range(len(cfg['candidates']))]) for region,row in frame.iterrows() if row.valid_n>0}


def map_view(a,b=None,mode='single',key='map',diff=None,zoom=True):
    focus=zoom and st.session_state.get('map_focus_'+country,False)
    selection=map_component(country=country,title=nice_region(st.session_state[region_key]) if focus else cfg['label'],geo=geo,colors=cfg['colors'],candidates=cfg['candidates'],selected=st.session_state[region_key] if focus else None,focus=focus,national=tx('national'),a=payload(a),b=payload(b) if b is not None else {},mode=mode,
        label_a=run_label(run_a) if mode in ('compare','animate') else run_label(run_b),label_b=run_label(run_b),diff=diff or {},limit=30,hint=tx('map_hint'),empty=tx('empty'),play=tx('play'),pause=tx('pause'),key=key,default=None)
    if selection and selection.get('sequence')!=st.session_state.get(key+'_event'):
        st.session_state[key+'_event']=selection['sequence']
        if selection.get('reset'):
            st.session_state['map_focus_'+country]=False
            st.session_state.pop('selected_persona',None)
            st.rerun()
        region=selection.get('region')
        if zoom and region in regions:
            st.session_state['pending_region']=region
            st.session_state['map_focus_'+country]=True
            st.session_state.pop('selected_persona',None)
            st.rerun()


ARM_COLORS={'demographic':'#57534e','cultural':'#0f5132','persona':'#5b3a8a','career':'#9a5b13'}


def evolution(pid,db):
    """Same persona across every linked context of the selected language, then the model's reason."""
    lang=RUNS[run_b]['language'];rows='';quotes=''
    for r in [k for k,v in RUNS.items() if v['country']==country and v['language']==lang and k not in UNLINKED]:
        d=load_detail(pid,r,version)
        if not d or not d['normalized']['valid']:continue
        vals,reason=response_of(d,cfg);w=leader(vals);arm=RUNS[r]['arm']
        if r in (run_a,run_b):quotes+=f'<div class="rq{" current" if r==run_b else ""}" style="--c:{ARM_COLORS[arm]}"><small>{esc(arm_label(arm))}</small><div class="quote">{esc(reason) or "—"}</div></div>'
        seg=''.join(f'<span style="width:{v}%;background:{cfg["colors"][i]};color:{on(cfg["colors"][i])}">{f"{v:.0f}%" if v>=12 else ""}</span>' for i,v in enumerate(vals))
        top=f'<span style="color:{ink(cfg["colors"][w])}">{esc(short_name(cfg["candidates"][w]))}</span>' if w>=0 else esc(tx('tie'))
        rows+=f'<tr class="{"current" if r==run_b else ""}"><td><span class="ctx" style="color:{ARM_COLORS[arm]}">{esc(arm_label(arm))}</span></td><td><div class="seg">{seg}</div></td><td class="top">{top}</td></tr>'
    left,right=st.columns([1.5,1],gap='large')
    with left:
        st.markdown(f'<div class="card"><div class="card-head"><div><h3>{esc(tx("evolution"))}</h3><p>{esc(tx("evolution_desc"))}</p></div></div><table class="evo"><thead><tr><th>{esc(g("context"))}</th><th>{esc(tx("probability"))}</th><th style="text-align:right">{esc(tx("top_choice"))}</th></tr></thead><tbody>{rows}</tbody></table></div>',unsafe_allow_html=True)
    with right:
        st.markdown(f'<div class="quote-card"><h3>{esc(tx("reason"))}</h3><div class="sub">{esc(tx("evolution_desc"))}</div>{quotes}</div>',unsafe_allow_html=True)


def ten_table(ids):
    """All sampled personas at once: A answer, B answer, status and both reasons."""
    if not ids:st.info(g('missing'));return
    show_key='ten_all_'+run_b
    def seg(vals):return '<div class="seg">'+''.join(f'<span style="width:{v}%;background:{cfg["colors"][i]};color:{on(cfg["colors"][i])}">{f"{v:.0f}%" if v>=14 else ""}</span>' for i,v in enumerate(vals))+'</div>'
    rows=[];changed=0
    for i,pid in enumerate(ids):
        da,db=load_detail(pid,run_a,version),load_detail(pid,run_b,version)
        if not (da and db and da['normalized']['valid'] and db['normalized']['valid']):continue
        (va,ra),(vb,rb)=response_of(da,cfg),response_of(db,cfg)
        moved=leader(va)!=leader(vb);changed+=moved
        p=db['profile'];region=p.get(cfg['region'],'');region=tu.STATE_NAME.get(region,region) if country=='usa' else region
        meta=' · '.join(esc(str(x).replace('_',' ')) for x in (f'#{len(rows)+1:02d}',p.get('age'),p.get('sex',p.get('gender'))) if x not in (None,''))
        job='' if country=='el_salvador' else esc(str(p.get('occupation') or '').replace('_',' '))
        status=f'<span class="status {"changed" if moved else "same"}">{esc(tx("changed_short") if moved else tx("same_short"))}</span>'
        reasons=f'<p title="{esc(ra)}"><b>A:</b>{esc(ra)}</p><p title="{esc(rb)}"><b>B:</b>{esc(rb)}</p>'
        rows.append(f'<div class="ten-row"><div class="ten-who"><img src="{robot(pid)}" alt="Robot"><div><b>{esc(region)}</b><small>{meta}</small><small>{job}</small></div></div>{seg(va)}<div class="ten-arrow">→</div>{seg(vb)}<div>{status}</div><div class="ten-reasons">{reasons}</div></div>')
    shown=rows if st.session_state.get(show_key) else rows[:5]
    head=f'<div class="ten-row head"><span>{esc(tx("profile_col"))}</span><span>A · {esc(run_label(run_a))}</span><span></span><span>B · {esc(run_label(run_b))}</span><span>{esc(tx("status"))}</span><span>{esc(tx("reasons_ab"))}</span></div>'
    foot=f'<div class="ten-foot"><span>{esc(tx("n_changed")).format(k=f"<b>{changed}</b>",n=len(rows))}</span></div>'
    st.markdown(f'<div class="ten">{head}{"".join(shown)}{foot}</div>',unsafe_allow_html=True)
    if len(rows)>5:
        label=tx('show_less') if st.session_state.get(show_key) else tx('show_all').format(n=len(rows))
        if st.button(label,key='toggle_'+show_key):st.session_state[show_key]=not st.session_state.get(show_key,False);st.rerun()


def profile_view(pid,full=False):
    da=load_detail(pid,run_a,version)
    db=load_detail(pid,run_b,version)
    available=db or da
    if not available:
        st.info(tx('no_response'));return
    p=available['profile']
    if full:
        region=p.get(cfg['region'],'');region=tu.STATE_NAME.get(region,region) if country=='usa' else region
        salvador=country=='el_salvador'  # census demographics redrawn there: show only prompt fields
        facts=[f'{tx("age")} {p.get("age")}' if p.get('age') not in (None,'') else '',p.get('sex',p.get('gender')),p.get('education_level',p.get('education')),None if salvador else p.get('occupation'),None if salvador else p.get('city') or p.get('municipality')]
        line=' · '.join(esc(str(f).replace('_',' ')) for f in facts if f not in (None,''))
        (hero_top if hero_top is not None else st).markdown(f'<div class="profile-hero"><img src="{robot(pid)}" alt="Robot"><div><h2>{esc(g("synthetic"))} · {esc(region)}</h2><div class="facts-line">{line}</div><div class="pid">{esc(pid)}</div></div></div>',unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="personahead"><img src="{robot(pid)}" alt="Robot"><div><b>{esc(pid)}</b><br><small>{esc(p.get("city",p.get("municipality",p.get(cfg["region"],""))))}</small></div></div>',unsafe_allow_html=True)
    if db and db['normalized']['valid'] and not full:
        response=db['raw']['response']
        st.caption(run_label(run_b))
        st.markdown('<div style="display:flex;height:10px;border-radius:8px;overflow:hidden;margin:8px 0">'+''.join(f'<span style="width:{float(response[c])}%;background:{cfg["colors"][i]}"></span>' for i,c in enumerate(cfg['candidates']))+'</div>',unsafe_allow_html=True)
        st.caption(' · '.join(f'{c}: {float(response[c]):.1f}%' for c in cfg['candidates']))
        reason=next((response[k] for k in ('reason','razon','razao') if k in response),'')
        if reason:
            st.markdown('**'+tx('reason')+'**')
            if full or len(str(reason))<=240:
                st.write(reason)
            else:
                st.write(str(reason)[:240].rsplit(' ',1)[0]+'…')
                with st.expander(tx('read_reason')):st.write(reason)
    if not full and st.button(tx('full'),key='full_'+pid):
        st.session_state['profile_page']=pid;st.query_params['persona']=pid;st.rerun()
    if not full:
        quick={tx('age'):p.get('age'),tx('sex'):p.get('sex',p.get('gender')),tx('education'):p.get('education_level',p.get('education')),tx('occupation'):p.get('occupation')}
        st.markdown('<div class="profile-grid">'+''.join('<div class="profile-field"><small>'+esc(k)+'</small><b>'+esc(str(v).replace('_',' '))+'</b></div>' for k,v in quick.items() if v is not None)+'</div>',unsafe_allow_html=True)
        return
    tabs=st.tabs([tx('result'),tx('profile'),tx('prompt'),'JSON'])
    with tabs[1]:
        fields={k:v for k,v in p.items() if k not in ('persona','cultural_background','persona_en','cultural_background_en','career_goals_and_ambitions') and '.' not in k}
        st.markdown('<div class="profile-grid">'+''.join('<div class="profile-field"><small>'+esc(k.replace('_',' '))+'</small><b>'+esc(str(v).replace('_',' '))+'</b></div>' for k,v in fields.items())+'</div>',unsafe_allow_html=True)
        with st.expander(tx('more')):
            for key in ('cultural_background','persona','cultural_background_en','persona_en','career_goals_and_ambitions'):
                if p.get(key): st.markdown('**'+key.replace('_',' ').title()+'**');st.write(p[key])
            ocean={k:v for k,v in p.items() if '.' in k}
            if ocean: st.json(ocean)
    with tabs[0]:
        evolution(pid,db)
        st.markdown('#### A → B')
        base=response_of(da,cfg)[0] if da and da['normalized']['valid'] else None
        for side,(column,run,data) in enumerate(zip(st.columns(2),(run_a,run_b),(da,db))):
            with column:
                if not (data and data['normalized']['valid']):st.info(tx('no_response'));continue
                vals,_=response_of(data,cfg);arm=RUNS[run]['arm']
                deltas=f'<div class="answer-choice">'+' · '.join(f'{esc(short_name(c))} <b>{vals[i]-base[i]:+.1f} pp</b>' for i,c in enumerate(cfg['candidates']))+'</div>' if side and base and run!=run_a else f'<div class="answer-choice">{esc(tx("baseline"))}</div>'
                st.markdown(f'<div class="answer{" b" if side else ""}" style="--c:{ARM_COLORS[arm]}"><div class="answer-label" style="color:{ARM_COLORS[arm]}">{"A" if not side else "B"} · {esc(run_label(run))}</div>{prob_html(vals,cfg)}{deltas}</div>',unsafe_allow_html=True)
    with tabs[2]:
        st.info('Current documented context — a historical exact prompt is not stored in these checkpoints. This is not a verified reconstruction of the submitted request.')
        st.caption('A / '+RUNS[run_a]['label']);st.json(current_context(run_a),expanded=False)
        st.caption('B / '+RUNS[run_b]['label']);st.json(current_context(run_b),expanded=False)
        fields_for={'demographic':'Demographics','cultural':'Demographics + cultural_background','persona':'Demographics + persona','career':'Demographics + career goals + Big Five'}
        st.write('A: '+fields_for[RUNS[run_a]['arm']]+' · B: '+fields_for[RUNS[run_b]['arm']])
        def prompt_text(run):
            r=RUNS[run];field=ADDED_FIELD.get(r['arm'])
            if field and r['language']=='en' and country!='usa' and field+'_en' in p:field+='_en'
            return json.dumps(current_context(run),ensure_ascii=False,indent=2)+'\n'+(str(p.get(field,'[Unavailable in source profile]')) if field else '')
        diff='\n'.join(difflib.unified_diff(prompt_text(run_a).splitlines(),prompt_text(run_b).splitlines(),fromfile='A: documented context/profile',tofile='B: documented context/profile',lineterm=''))
        st.code(diff or 'No difference in the displayed source text.',language='diff')
        st.caption('Source text differences; exact historical rendering unavailable. Identical configuration does not control every possible experimental difference.')
    with tabs[3]:
        for data,run in [(da,run_a),(db,run_b)]:
            if data:
                st.markdown('**'+RUNS[run]['label']+'**');st.json(data,expanded=False)
        blob=json.dumps({'persona_id':pid,'A':da,'B':db},ensure_ascii=False,default=str,indent=2)
        st.download_button('Download JSON',blob,file_name=pid.replace(':','_')+'.json',mime='application/json',key='download_full')


def persona_list(frame):
    region=st.session_state[region_key]
    rows=frame.loc[frame.region.eq(region)].copy()
    pid=st.session_state.get('selected_persona')
    if pid and pid in set(rows.pid):
        if st.button('← '+nice_region(region),key='back_people'):
            st.session_state.pop('selected_persona',None);st.rerun()
        profile_view(pid)
        return
    page_key='sample_page'+country;page=st.session_state.get(page_key,0)
    with st.container(key='territory_head'):
        head,shuffle=st.columns([4,1],vertical_alignment='center')
        head.subheader(nice_region(region))
        if shuffle.button('',icon=':material/casino:',key='next_sample',help=tx('next'),width='stretch'):st.session_state[page_key]=page+1;st.rerun()
    st.markdown(f'<div class="territory-count"><b>{len(rows):,}</b> {esc(tx("agents").lower())}</div>',unsafe_allow_html=True)
    invalid_n=int((rows.valid==0).sum())
    if region in summary_b.index and summary_b.loc[region,'valid_n']>0:
        values=summary_b.loc[region,[f'p{i}' for i in range(len(cfg['candidates']))]].to_numpy(dtype=float)
        top=leader(values)
        st.caption(tx('sim_leader')+': '+(cfg['candidates'][top] if top>=0 else tx('tie')))
        if country=='usa':
            st.caption(f'{tu.EV_2024[region]} '+tx('electoral_votes'))
        st.markdown(''.join(f'<div class="result-row"><span>{esc(c)}</span><b>{summary_b.loc[region,f"p{i}"]:.1f}%</b></div><div class="result-track"><span style="width:{summary_b.loc[region,f"p{i}"]:.1f}%;background:{cfg["colors"][i]}"></span></div>' for i,c in enumerate(cfg['candidates'])),unsafe_allow_html=True)
    if rows.empty:st.info(tx('empty'));return
    offset=(page*3)%len(rows)
    rows['_sample_order']=pd.util.hash_pandas_object(rows.pid,index=False).values
    sample=rows.sort_values('_sample_order').iloc[offset:offset+3]
    for _,row in sample.iterrows():
        winner=int(row.winner)
        choice=cfg['candidates'][winner] if winner>=0 and row.valid else tx('tie') if row.valid else tx('invalid')
        color=cfg['colors'][winner] if winner>=0 and row.valid else '#5b6670'
        with st.container(border=True):
            age=f'{row.age:g}' if pd.notna(row.age) else '—'
            vals=[float(row[f'p{i}']) for i in range(len(cfg['candidates']))] if row.valid else []
            split=('<div class="split">'+''.join(f'<span style="width:{v}%;background:{cfg["colors"][i]}"></span>' for i,v in enumerate(vals))+'</div><div class="split-nums">'+''.join(f'<span style="color:{ink(cfg["colors"][i])}">{v:.0f}% {esc(short_name(cfg["candidates"][i]))}</span>' for i,v in enumerate(vals))+'</div>') if vals else ''
            who,go=st.columns([5,1],vertical_alignment='center')
            who.markdown(f'<div class="compact-person"><img src="{robot(row.pid)}" alt="Robot"><div class="who"><strong>{esc(row.city or nice_region(row.region))}</strong><small>{age} · {esc(row.sex)} · <span style="color:{ink(color)}">{esc(choice)}</span></small></div></div>',unsafe_allow_html=True)
            if go.button('',icon=':material/arrow_outward:',key='open'+row.pid,type='primary',help=tx('open'),width='stretch'):
                st.session_state['profile_page']=row.pid;st.query_params['persona']=row.pid;st.rerun()
            if split:st.markdown(split,unsafe_allow_html=True)


def full_profile_page():
    pid=st.session_state.get('profile_page') or st.query_params.get('persona')
    if not pid: return False
    if not pid.startswith(country+':'):
        st.session_state.pop('profile_page',None);st.query_params.clear();return False
    profile_view(pid,True)
    return True




# ---------- start panel: three example states (USA) ----------

def example_picks():
    """Chosen by rule from the real 2024 result (DC excluded): largest Trump margin, largest Harris margin, closest race."""
    margins={s:t-h for s,(_,t,h) in tu.ELECTION_2024.items() if s!='DC'}
    return [('rep',max(margins,key=margins.get)),('dem',min(margins,key=margins.get)),('swing',min(margins,key=lambda s:abs(margins[s])))]


def example_states(frame):
    st.markdown(f'<div class="territory-intro compact"><h3>{esc(tx("examples_title"))}</h3><p>{esc(tx("examples_desc"))}</p></div>',unsafe_allow_html=True)
    for kind,state in example_picks():
        _,t,h=tu.ELECTION_2024[state]
        accent={'rep':cfg['colors'][0],'dem':cfg['colors'][1],'swing':'#8a6d1d'}[kind]
        name=tu.STATE_NAME[state]
        tip=esc(tx("ex_"+kind+"_tip"))
        sim=''
        if state in summary_b.index and summary_b.loc[state,'valid_n']>0:
            s0,s1=float(summary_b.loc[state,'p0']),float(summary_b.loc[state,'p1'])
            sim=f'<span>{esc(tx("sim_short"))} <b style="color:{ink(cfg["colors"][0])}">{s0:.0f}</b>–<b style="color:{ink(cfg["colors"][1])}">{s1:.0f}</b></span>'
        rows=frame.loc[frame.region.eq(state)&frame.valid.eq(1)&frame.winner.ge(0)]
        person='';pid=None
        if not rows.empty:
            rows=rows.assign(_o=pd.util.hash_pandas_object(rows.pid,index=False).values).sort_values('_o')
            r=rows.iloc[0];pid=r.pid;w=int(r.winner)
            vals=[float(r.p0),float(r.p1)]
            facts=' · '.join(esc(str(x)) for x in (r.city or name,f'{r.age:g}' if pd.notna(r.age) else None,r.sex) if x not in (None,''))
            person=f'<div class="ex-person"><img src="{robot(pid)}" alt="Robot"><small>{facts}</small><b style="color:{ink(cfg["colors"][w])}">{esc(short_name(cfg["candidates"][w]))} {vals[w]:.0f}%</b></div>'
        with st.container(key='ex_'+kind):
            st.markdown(f'<div class="ex-card" style="--accent:{accent}"><div class="ex-head"><span class="ex-kind" style="color:{ink(accent)}">{esc(tx("ex_"+kind))}</span><span class="tip" tabindex="0" role="note" aria-label="{tip}" data-tip="{tip}">i</span></div><div class="ex-state">{esc(name)}</div><div class="ex-line"><span>{esc(tx("real_2024"))} <b style="color:{ink(cfg["colors"][0])}">{t:.0f}</b>–<b style="color:{ink(cfg["colors"][1])}">{h:.0f}</b></span>{sim}</div>{person}</div>',unsafe_allow_html=True)
            left,right=st.columns(2,gap='small')
            if pid and left.button(tx('open')+' →',key='ex_open_'+state,width='stretch'):
                st.session_state['profile_page']=pid;st.query_params['persona']=pid;st.rerun()
            if right.button(tx('explore_short')+' →',key='ex_go_'+state,width='stretch'):
                st.session_state['pending_region']=state;st.session_state['map_focus_usa']=True;st.session_state.pop('selected_persona',None);st.rerun()


# ---------- the data: how the synthetic voters were built ----------

DATA_REFS={
    'usa':('usa_demographic_en',(('sex',tu.US_SEX_REFERENCE),('age_group',tu.US_AGE_18PLUS))),
    'el_salvador':(None,(('department',{k:v/sum(tu.SV_POPULATION_BY_DEPARTMENT.values()) for k,v in tu.SV_POPULATION_BY_DEPARTMENT.items()}),('age_group',tu.SV_AGE_18PLUS),('sex',tu.SV_SEX_18PLUS))),
    'brazil':('brazil_demographic_pt',(('region',{k:v/sum(tu.BR_POPULATION_2022.values()) for k,v in tu.BR_POPULATION_2022.items()}),('sex',tu.BR_SEX_2022))),
}


@st.cache_data(show_spinner=False)
def input_vs_census(country,version):
    """Personas sent to the model vs official shares, same method as 01_data.ipynb (TVD = half the sum of absolute gaps)."""
    run,refs=DATA_REFS[country]
    if country=='el_salvador':
        pre=APP_DIR/'generated'/'sv_census.json'  # cloud bundle: precomputed, the parquet stays local
        if pre.exists():
            j=json.loads(pre.read_text(encoding='utf-8'))
            return {var:dict(table=pd.DataFrame({'input':d['table']['input'],'official':d['table']['official'],'diff':d['table']['diff']},index=d['table']['index']),tvd=d['tvd']) for var,d in j.items()}
        people=tu.load_sv_people(['department','age_group','sex']).astype(str);weight=None
    else:
        people=load_run(run,version)[['region','sex','age']].copy()
        people['age_group']=tu.age_group(people.age).astype(str)
        weight=None
        if country=='usa':  # Kish over-samples close states on purpose: re-weight to state population first
            share=pd.Series(tu.POPULATION_2025)/sum(tu.POPULATION_2025.values())
            weight=people.region.map(share/people.region.value_counts(normalize=True))
    out={}
    for var,ref in refs:
        ref=pd.Series(ref,dtype=float)
        obs=(weight.groupby(people[var]).sum() if weight is not None else people[var].value_counts()).astype(float)
        obs=(obs/obs.sum()).reindex(ref.index,fill_value=0)
        table=pd.DataFrame({'input':obs*100,'official':ref*100});table['diff']=table.input-table.official
        out[var]=dict(table=table,tvd=float((obs-ref).abs().sum()/2))
    return out


DATA_ICONS=['<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><circle cx="12" cy="7" r="3.2"/><path d="M5.5 20c.8-3.6 3.4-5.6 6.5-5.6s5.7 2 6.5 5.6"/><path d="M19 4.5l1.2 1.2M4.8 4.5 3.6 5.7"/></svg>',
            '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"><path d="M3 6l6-2 6 2 6-2v14l-6 2-6-2-6 2z"/><path d="M9 4v14M15 6v14"/></svg>',
            '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><path d="M4 19V9M10 19V5M16 19v-7M22 19H2"/></svg>']


def data_view():
    st.markdown(f'<span class="pill">NVIDIA NeMo Data Designer</span><div class="section-head"><h2>{esc(tx("data_title"))}</h2><p>{esc(tx("data_intro"))}</p></div>',unsafe_allow_html=True)
    steps=''.join(f'<div class="data-step"><div class="icon">{DATA_ICONS[i]}</div><small>{i+1:02d}</small><b>{esc(tx(f"data_s{i+1}_t"))}</b><p>{esc(tx(f"data_s{i+1}"))}</p></div>' for i in range(3))
    st.markdown(f'<div class="data-steps">{steps}</div>',unsafe_allow_html=True)
    st.markdown(f'<div class="section-head small"><h3>{esc(tx("data_compare"))}</h3></div>',unsafe_allow_html=True)
    which=st.radio(tx('data_compare'),list(DATA_REFS),format_func=lambda k:COUNTRIES[k]['label'],horizontal=True,key='seg_data_country',label_visibility='collapsed')
    result=input_vs_census(which,version)
    st.markdown(f'<div class="note">{esc(tx("src_"+which))}</div>',unsafe_allow_html=True)
    cards=''
    for var,res in result.items():
        table=res['table'];top=max(table.input.max(),table.official.max()) or 1
        label=lambda k:tu.STATE_NAME.get(k,k) if which=='usa' else k
        rows=''.join(f'<div class="dq-row"><span class="dq-cat">{esc(label(k))}</span><div class="dq-bars"><i class="in" style="width:{r.input/top*100:.1f}%"></i><i class="off" style="width:{r.official/top*100:.1f}%"></i></div><span class="dq-num">{r.input:.1f}%<small>{r.official:.1f}%</small></span><span class="dq-diff">{r["diff"]:+.2f}</span></div>' for k,r in table.iterrows())
        cards+=f'<div class="dq-card"><div class="card-head"><div><h3>{esc(tx("var_"+var))}</h3></div><span class="tvd">TVD {res["tvd"]:.4f}</span></div><div class="dq-legend"><span><i class="in"></i>{esc(tx("col_input"))}</span><span><i class="off"></i>{esc(tx("col_official"))}</span><span>{esc(tx("col_diff"))}</span></div>{rows}</div>'
    st.markdown(f'<div class="dq-grid">{cards}</div>',unsafe_allow_html=True)
    if which=='el_salvador':st.markdown(f'<div class="alert-invalid"><b>{esc(tx("sv_caveat_t"))}</b>{esc(tx("sv_caveat"))}</div>',unsafe_allow_html=True)
    if which=='usa':st.caption(tx('usa_state_note'))
    csv=pd.concat([res['table'].rename_axis('category').reset_index().assign(country=which,variable=var,tvd=res['tvd']) for var,res in result.items()])[['country','variable','category','input','official','diff','tvd']].rename(columns={'input':'input_pct','official':'official_pct','diff':'diff_pp'}).round(4).to_csv(index=False)
    st.download_button(tx('download_csv')+' · '+COUNTRIES[which]['label'],csv,file_name=f'personas_vs_census_{which}.csv',mime='text/csv',key='csv_'+which)
    # Kish allocation (data/allocation.csv, the allocation actually used; see 01_data.ipynb and THESIS_LOG.md §3)
    alloc=pd.read_csv(ROOT/'data'/'allocation.csv').sort_values('n_calls',ascending=False)
    alloc['kish']=alloc.n_calls/alloc.n_calls.sum()*100;alloc['pop_share']=alloc['pop']/alloc['pop'].sum()*100
    st.markdown(f'<div class="section-head small"><h3>{esc(tx("kish_title"))}</h3><p>{esc(tx("kish_text"))}</p></div>',unsafe_allow_html=True)
    stats=[('1.17',tx('kish_deff')),('170,911',tx('kish_eff')),('200 – 50%',tx('kish_box')),('200,023',tx('kish_total'))]
    st.markdown('<div class="mini-stats">'+''.join(f'<div><b>{esc(v)}</b><small>{esc(k)}</small></div>' for v,k in stats)+'</div>',unsafe_allow_html=True)
    fig=go.Figure([go.Bar(x=alloc.state,y=alloc.kish,name=tx('kish_series'),marker_color='#0f5132'),go.Bar(x=alloc.state,y=alloc.pop_share,name=tx('kish_pop'),marker_color='#b9ad99')])
    fig.update_layout(barmode='group',height=340,margin=dict(l=10,r=10,t=10,b=10),paper_bgcolor='#ffffff',plot_bgcolor='#ffffff',legend=dict(orientation='h',y=1.1,x=0),yaxis=dict(title='%',gridcolor='#efebe4'),xaxis=dict(tickfont=dict(size=10)),font=dict(family='Inter',color='#1c1917'))
    st.plotly_chart(fig,width='stretch',key='kish_chart',config={'displayModeBar':False})
    st.caption(tx('kish_note'))
    st.markdown(f'<div class="callout">{esc(tx("data_takeaway"))}</div>',unsafe_allow_html=True)




# ---------- how it ran ----------

PIPE_ICONS=['<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M3 9h18M3 14h18M9 4v16"/></svg>',
            '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"><path d="M12 3l8 4.5v9L12 21l-8-4.5v-9z"/><path d="M12 12l8-4.5M12 12v9M12 12L4 7.5"/></svg>',
            '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"><rect x="2" y="7" width="20" height="11" rx="2"/><circle cx="8" cy="12.5" r="3"/><circle cx="16" cy="12.5" r="3"/><path d="M4 18v2M20 18v2"/></svg>',
            '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M4 12.5l5 5L20 6.5"/></svg>']
GPU_ART='<svg class="gpu-art" viewBox="0 0 320 150" fill="none"><rect x="10" y="30" width="290" height="96" rx="10" fill="#1c1917"/><rect x="10" y="30" width="290" height="96" rx="10" stroke="#3a3531" stroke-width="2"/><rect x="24" y="44" width="262" height="68" rx="8" fill="#2a2622"/><circle cx="92" cy="78" r="26" fill="#0f5132"/><circle cx="92" cy="78" r="18" fill="#1c1917"/><circle cx="92" cy="78" r="6" fill="#95d4ac"/><circle cx="196" cy="78" r="26" fill="#0f5132"/><circle cx="196" cy="78" r="18" fill="#1c1917"/><circle cx="196" cy="78" r="6" fill="#95d4ac"/><rect x="40" y="126" width="120" height="8" rx="2" fill="#b9ad99"/><rect x="250" y="52" width="24" height="52" rx="4" fill="#3a3531"/><text x="160" y="24" text-anchor="middle" font-family="Inter,sans-serif" font-size="12" font-weight="600" fill="#57534e">RTX A6000 · 48 GB · 0,53 $/h</text></svg>'


def pipeline_view():
    logo=APP_DIR/'assets'/'vllm-logo.png'
    vllm=f'<img class="vllm-logo" alt="vLLM" src="data:image/png;base64,{base64.b64encode(logo.read_bytes()).decode()}">' if logo.is_file() else '<b class="vllm-text">vLLM</b>'
    st.markdown(f'<span class="pill">personasurvey · vLLM · RunPod</span><div class="section-head"><h2>{esc(tx("pipe_title"))}</h2><p>{esc(tx("pipe_intro"))}</p></div>',unsafe_allow_html=True)
    steps=''.join(f'<div class="data-step"><div class="icon">{PIPE_ICONS[i]}</div><small>{i+1:02d}</small><b>{esc(tx(f"pipe_s{i+1}_t"))}</b><p>{esc(tx(f"pipe_s{i+1}"))}</p></div>' for i in range(4))
    st.markdown(f'<div class="data-steps four">{steps}</div>',unsafe_allow_html=True)
    chips=''.join(f'<span class="chip">{esc(c)}</span>' for c in tx('ps_chips').split('|'))
    st.markdown(f'<div class="card ps-card"><div class="card-head"><div><h3>{esc(tx("ps_title"))}</h3><p>{esc(tx("ps_sub"))}</p></div><a class="tvd" href="https://github.com/Japulgarin/personasurvey" target="_blank" rel="noopener">GitHub · MIT</a></div><p class="card-text">{esc(tx("ps_text"))}</p><div class="chips">{chips}</div></div>',unsafe_allow_html=True)
    st.markdown(f'<div class="section-head small"><h3>{esc(tx("gpu_title"))}</h3><p>{esc(tx("gpu_text"))}</p></div>',unsafe_allow_html=True)
    art,stats=st.columns([1,1.6],gap='large',vertical_alignment='center')
    with art:st.markdown(f'<div class="gpu-card">{GPU_ART}<div class="gpu-serve">{vllm}<span>{esc(tx("gpu_serve"))}</span></div></div>',unsafe_allow_html=True)
    with stats:
        rows=[('0,53 $/h',tx('gpu_s1')),('2',tx('gpu_s2')),('~20/s',tx('gpu_s3')),('600',tx('gpu_s4')),('~2,8 h',tx('gpu_s5')),('~3 $',tx('gpu_s6'))]
        st.markdown('<div class="mini-stats six">'+''.join(f'<div><b>{esc(v)}</b><small>{esc(k)}</small></div>' for v,k in rows)+'</div>',unsafe_allow_html=True)
    st.markdown(f'<div class="note"><b>{esc(tx("limits_title"))}.</b>&nbsp;{esc(tx("limits_text"))}</div>',unsafe_allow_html=True)
    # TranslateGemma (03_translation.ipynb)
    st.markdown(f'<div class="section-head small"><h3>{esc(tx("tr_title"))}</h3><p>{esc(tx("tr_text"))}</p></div>',unsafe_allow_html=True)
    tr=[('98.3%',tx('tr_sv')),('99.6%',tx('tr_br')),('100%',tx('tr_final')),('43.7 min',tx('tr_time'))]
    st.markdown('<div class="mini-stats">'+''.join(f'<div><b>{esc(v)}</b><small>{esc(k)}</small></div>' for v,k in tr)+'</div>',unsafe_allow_html=True)
    st.markdown(f'<div class="note">{esc(tx("tr_lesson"))}</div>',unsafe_allow_html=True)
    # Cost: from the tokens actually processed (06_summary_costs.ipynb prices and throughput rule)
    st.markdown(f'<div class="section-head small"><h3>{esc(tx("cost_title"))}</h3><p>{esc(tx("cost_text"))}</p></div>',unsafe_allow_html=True)
    cs=json.loads((APP_DIR/'generated'/'cost_summary.json').read_text(encoding='utf-8'))
    agg={}
    for r in cs.values():
        a=agg.setdefault(r['country'],dict(requests=0,in_tok=0,out_tok=0))
        for k in a:a[k]+=r[k]
    agg['total']={k:sum(a[k] for a in agg.values()) for k in ('requests','in_tok','out_tok')}
    def costs(a):
        api=a['in_tok']/1e6*0.05+a['out_tok']/1e6*0.10;hours=a['requests']/20/3600;return api,hours,hours*2*0.53
    rows=''
    for key,a in agg.items():
        api,hours,gpu=costs(a);label=COUNTRIES[key]['label'] if key in COUNTRIES else tx('cost_total')
        rows+=f'<tr class="{"total" if key=="total" else ""}"><td>{esc(label)}</td><td>{a["requests"]:,}</td><td>{a["in_tok"]/1e6:,.1f} M</td><td>{a["out_tok"]/1e6:,.1f} M</td><td>${api:,.2f}</td><td>{hours:.1f} h</td><td>${gpu:,.2f}</td></tr>'
    st.markdown(f'<div class="card"><table class="cost-table"><thead><tr><th>{esc(tx("cost_country"))}</th><th>{esc(tx("cost_answers"))}</th><th>{esc(tx("cost_in"))}</th><th>{esc(tx("cost_out"))}</th><th>{esc(tx("cost_api"))}</th><th>{esc(tx("cost_gpu_time"))}</th><th>{esc(tx("cost_gpu"))}</th></tr></thead><tbody>{rows}</tbody></table></div>',unsafe_allow_html=True)
    api,hours,gpu=costs(agg['total'])
    st.markdown('<div class="mini-stats"><div><b>$'+f'{api:,.0f}'+'</b><small>'+esc(tx('cost_api_total'))+'</small></div><div><b>$'+f'{gpu:,.0f}'+'</b><small>'+esc(tx('cost_gpu_total'))+'</small></div><div><b>'+f'{api/gpu:.1f}×'+'</b><small>'+esc(tx('cost_saving'))+'</small></div><div><b>'+f'{hours:.0f} h'+'</b><small>'+esc(tx('cost_hours'))+'</small></div></div>',unsafe_allow_html=True)
    st.caption(tx('cost_note'))


# ---------- research: stability tests ----------

def stability_view():
    st.markdown(f'<div class="section-head"><h2>{esc(tx("lang_test"))}</h2><p>{esc(tx("lang_test_desc"))}</p></div>',unsafe_allow_html=True)
    with st.expander(tx('what_is')):st.write(tx('lang_explain'))
    exp=st.radio(tx('lang_test'),['el_salvador','brazil'],format_func=lambda k:COUNTRIES[k]['label']+' · '+LOCAL_LANGUAGE[k].upper()+' → EN',horizontal=True,key='seg_lang_country',label_visibility='collapsed')
    c=COUNTRIES[exp];local=LOCAL_LANGUAGE[exp]
    rows=language_stats(exp,version)
    body=''
    for r in rows:
        means=''.join(f'<div class="prob-row"><span style="color:{ink(c["colors"][i])}">{esc(name)}</span><b>{r["local"][i]:.1f}% → {r["en"][i]:.1f}%</b></div>' for i,name in enumerate(c['candidates']))
        body+=f'<div class="lang-row"><div><b>{esc(tx(r["arm"]))}</b><small>{r["n"]:,} {esc(tx("matched").lower())}</small></div><div><span class="big">{r["changed"]:.1f}%</span><small>{esc(tx("lang_changed"))}</small></div><div><small>{esc(tx("lang_means"))}</small>{means}</div></div>'
    st.markdown(f'<div class="bench">{body}</div>',unsafe_allow_html=True)
    st.markdown('#### '+tx('lang_examples'))
    arm=st.radio(tx('variant'),['demographic','cultural','persona'],format_func=tx,horizontal=True,key='seg_lang_arm')
    ids=sample_ids((f'{exp}_{arm}_{local}',f'{exp}_{arm}_en'),version)
    if ids:
        pid=ids[st.session_state.get('lang_person_'+exp,0)%len(ids)]
        compare_pair(pid,f'{exp}_{arm}_{local}',f'{exp}_{arm}_en',c,tx(arm)+' · '+local.upper(),tx(arm)+' · EN',added=False)
        person_nav(ids,'lang_person_'+exp)
    else:st.info(g('missing'))
    st.divider()
    st.markdown(f'<div class="section-head"><h2>{esc(tx("swap_test"))}</h2><p>{esc(tx("swap_desc"))}</p></div>',unsafe_allow_html=True)
    with st.expander(tx('what_is')):st.write(tx('swap_explain'))
    st.caption(tx('swap_hypotheses'))
    lula,flavio=tu.BR_CANDIDATES
    def line(programme,label,bad=False): return f'<div class="swap-line{" bad" if bad else ""}"><span>{esc(tx("programme_of"))} {esc(programme)}</span><span>{esc(tx("labelled"))} “{esc(label)}”</span></div>'
    s=swap_stats()
    if not s:
        st.info(tx('swap_pending'))
    else:
        cols=st.columns(len(s))
        for col,(lang,r) in zip(cols,s.items()):
            name_wins=r['keep']>50
            col.markdown(f'<div class="swap-col"><h4>{esc(tx("swap_lang_"+lang))}</h4>'
                         f'<div class="swap-line"><span>{esc(tx("swap_keep_name"))}</span><span><b>{r["keep"]:.1f}%</b></span></div>'
                         f'<div class="swap-line"><span>{esc(tx("swap_follow_programme"))}</span><span><b>{r["follow"]:.1f}%</b></span></div>'
                         f'<div class="swap-line"><span>{esc(tx("swap_pairs"))}</span><span>{r["n"]:,}</span></div>'
                         f'<div class="swap-line"><span>{esc(tx("swap_lula_share"))}</span><span>{r["lula_o"]:.1f}% → {r["lula_s"]:.1f}%</span></div>'
                         f'<div class="swap-line"><span>{esc(tx("swap_pvalue"))}</span><span>{r["p"]:.3g}</span></div>'
                         f'<p class="swap-verdict">{esc(tx("swap_verdict_name" if name_wins else "swap_verdict_programme"))}</p></div>',unsafe_allow_html=True)
        st.caption(tx('swap_reading'))
    with st.expander(tx('swap_first_run')):
        st.write(tx('swap_bug'))
        intended=line('Lula',flavio)+line('Flávio',lula)
        recorded=line('Lula','Lula')+line('Flávio',lula,True)+f'<div class="swap-line bad"><span>{esc(flavio)}</span><span>{esc(tx("no_programme"))}</span></div>'
        st.markdown(f'<div class="swap-grid"><div class="swap-col"><h4>{esc(tx("swap_intended"))}</h4>{intended}</div><div class="swap-col"><h4>{esc(tx("swap_recorded"))}</h4>{recorded}</div></div>',unsafe_allow_html=True)


# ---------- about ----------

CREDITS=[
    ('cr_models',[
        ('NVIDIA Nemotron-Personas','https://huggingface.co/nvidia/datasets','cr_personas'),
        ('NVIDIA NeMo Data Designer','https://docs.nvidia.com/nemo/datadesigner/','cr_designer'),
        ('OpenAI gpt-oss-20b','https://huggingface.co/openai/gpt-oss-20b','cr_model'),
        ('Google TranslateGemma 4B','https://huggingface.co/google/translategemma-4b-it','cr_translate'),
        ('sentence-transformers · all-MiniLM-L6-v2','https://www.sbert.net/','cr_embed'),
        ('vLLM','https://github.com/vllm-project/vllm','cr_serve')]),
    ('cr_software',[
        ('Python · pandas · NumPy · SciPy · pyarrow','https://pandas.pydata.org/','cr_dataproc'),
        ('personasurvey','https://github.com/Japulgarin/personasurvey','cr_sim'),
        ('Streamlit','https://streamlit.io/','cr_app'),
        ('Plotly · Matplotlib','https://plotly.com/python/','cr_charts'),
        ('D3.js','https://d3js.org/','cr_maps_lib'),
        ('Shapely','https://shapely.readthedocs.io/','cr_geometry'),
        ('DiceBear','https://www.dicebear.com/','cr_avatars'),
        ('Google Fonts','https://fonts.google.com/','cr_fonts'),
        ('Noto Color Emoji','https://github.com/googlefonts/noto-emoji','cr_emoji')]),
    ('cr_data',[
        ('U.S. Census Bureau','https://www.census.gov/','cr_census_us'),
        ('National Archives · Electoral College','https://www.archives.gov/electoral-college/allocation','cr_results_us'),
        ('ONEC · TSE El Salvador','https://www.tse.gob.sv/','cr_census_sv'),
        ('IBGE','https://www.ibge.gov.br/','cr_census_br'),
        ('geoBoundaries','https://www.geoboundaries.org/','cr_boundaries'),
        ('Wikimedia Commons','https://commons.wikimedia.org/','cr_portraits'),
        ('Leslie Kish, Multipurpose sample designs','https://doi.org/10.1002/9781118150481','cr_kish')]),
    ('cr_services',[
        ('RunPod','https://www.runpod.io/','cr_gpu'),
        ('Cloudflare Tunnel','https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/','cr_tunnel'),
        ('Google Stitch','https://stitch.withgoogle.com/','cr_design'),
        ('Claude Code · Claude Fable 5.1 (Anthropic)','https://claude.com/claude-code','cr_assistant')]),
]


def credits_view():
    groups=''.join(f'<div class="credit-group"><h4>{esc(tx(key))}</h4><ul>'+''.join(f'<li><a href="{url}" target="_blank" rel="noopener">{esc(name)}</a><span>{esc(tx(note))}</span></li>' for name,url,note in items)+'</ul></div>' for key,items in CREDITS)
    st.markdown(f'<div class="credits"><div class="section-head"><h2>{esc(tx("credits"))}</h2><p>{esc(tx("credits_intro"))}</p></div><div class="credit-grid">{groups}</div></div>',unsafe_allow_html=True)


def about_view():
    main,side=st.columns([2.2,1],gap='large')
    with side:
        st.markdown('<div class="fact-sheet"><h4>'+esc(g('fact_sheet'))+'</h4>'+''.join(f'<small>{esc(t)}</small><b>{esc(n)}</b>' for t,n in zip(g('pillars'),g('pillar_notes')))+'</div>',unsafe_allow_html=True)
    with main:
        st.markdown(f'<span class="pill">{esc(g("thesis_tag"))}</span><div class="section-head"><h2>{esc(g("questions"))}</h2></div><div class="questions">'+''.join(f'<div class="question"><span>{i+1:02d}</span><p>{esc(h)}</p></div>' for i,h in enumerate(g('hypotheses')))+'</div>',unsafe_allow_html=True)
        st.markdown(f'<div class="callout">{esc(g("findings"))}</div>',unsafe_allow_html=True)
        about={
        'en':'This master’s thesis studies how synthetic NVIDIA Nemotron personas respond to electoral questions using gpt-oss-20b. It compares demographic profiles with cultural background and richer persona descriptions, across USA 2024, El Salvador 2024, and a Brazil 2026 scenario. The question is whether a plausible national total also contains plausible geographic and individual variation.',
        'es':'Esta tesis de maestría estudia cómo personas sintéticas de NVIDIA Nemotron responden a preguntas electorales con gpt-oss-20b. Compara perfiles demográficos, contexto cultural y descripciones de persona en Estados Unidos 2024, El Salvador 2024 y un escenario de Brasil 2026. Un total nacional plausible no garantiza patrones geográficos e individuales plausibles.',
        'pt':'Esta dissertação estuda como personas sintéticas NVIDIA Nemotron respondem a perguntas eleitorais com gpt-oss-20b. Compara perfis demográficos, contexto cultural e descrições de persona nos EUA 2024, El Salvador 2024 e num cenário do Brasil 2026. Um total nacional plausível não garante padrões geográficos e individuais plausíveis.',
        'de':'Diese Masterarbeit untersucht, wie synthetische NVIDIA-Nemotron-Personen mit gpt-oss-20b auf Wahlfragen antworten. Sie vergleicht demografische Profile, kulturellen Hintergrund und Personenbeschreibungen für die USA 2024, El Salvador 2024 und ein Brasilien-Szenario 2026. Plausible nationale Ergebnisse garantieren keine plausiblen regionalen oder individuellen Muster.'}
        st.write(about[st.session_state.lang])
        st.markdown('**200,023 USA · 50,000 El Salvador · 50,000 Brazil** — documented input personas.')
        with st.expander(g('deep')):
            st.markdown((ROOT/'RESEARCH_QUESTIONS.md').read_text(encoding='utf-8'))
        with st.expander(g('methods')):
            st.markdown('The main completed conditions are available here. Brazil partial full-context runs and the invalid label-swap run are excluded from the maps; the label swap is documented under Extra experiments. No new model calls are made.')
            st.markdown('**Metrics.** Mean probability averages each candidate’s stated probability. Top-choice share counts unique largest probabilities; tied personas remain in the denominator and count toward no candidate. Sankey nodes include ties. USA national summaries use the project’s population weights. A/B map comparisons use valid matched identities only.')
            st.markdown('**Prompt provenance.** Checkpoints retain the response and generation configuration but generally not the complete submitted prompt. The prompt tab displays the current documented context, clearly identified as such.')
            st.markdown('**Palette.** Party-associated chart colors are design approximations, not certified brand hex values. [GOP](https://shop.gop.com/collections/republican-national-committee) · [Democrats](https://democrats.org/) · [TSE El Salvador](https://tse.gob.sv/publico/prensa/330) · [TSE Brazil](https://www.tse.jus.br/partidos).')
            st.dataframe(query('SELECT * FROM runs'),hide_index=True,width='stretch')
            licenses={'CC BY-SA 4.0':'https://creativecommons.org/licenses/by-sa/4.0/','CC BY 2.0':'https://creativecommons.org/licenses/by/2.0/'}
            for candidate,asset in portraits.items():
                license=asset.get('license','')
                terms=f'[{license}]({licenses[license]})' if license in licenses else license
                st.markdown(f'**{candidate}** — [{asset.get("credit","Source")}]({asset.get("source","")}) · {terms}')
            st.caption('Portraits displayed with CSS crops; downloaded originals are unchanged. Robot avatars: DiceBear bottts-neutral 10.x.')
    credits_view()


def footer(): st.markdown(f'<div class="app-foot"><span><b>{esc(TITLE)}</b> · {esc(AUTHOR)} · {esc(tx("disclaimer"))}</span><span>gpt-oss-20b · NVIDIA Nemotron</span></div>',unsafe_allow_html=True)


# ---------- page ----------

if not DB.exists():
    st.info('The one-time index is being prepared. Refresh after the build finishes.')
    st.code('py tools/prepare_explorer_data.py');st.stop()
version=DB.stat().st_mtime_ns
portrait_path=APP_DIR/'assets'/'candidates'/'manifest.json'
portraits=json.loads(portrait_path.read_text(encoding='utf-8')) if portrait_path.exists() else {}

if not st.session_state.get('guide_done'):
    guide();st.stop()

profile_pid=st.session_state.get('profile_page') or st.query_params.get('persona')
if profile_pid:  # person page: no masthead or tabs, just a compact control row
    view='election'
else:
    topbar=st.container(key='topbar')
    with topbar:
      head,guidecol,langcol=st.columns([3,1,2.4],vertical_alignment='center')
      with head:st.markdown('<div class="masthead"><h1>'+esc(TITLE)+'</h1><p>'+esc(tx('app_subtitle'))+'</p></div>',unsafe_allow_html=True)
      with guidecol:st.button(g('reopen'),on_click=open_guide,width='stretch')
      with langcol:language_widget('interface_language')
      view=st.radio('View',['election','changes','experiments','data','pipeline','about'],format_func=tx,horizontal=True,label_visibility='collapsed',key='main_nav')

if view=='about':
    about_view();footer();st.stop()
if view=='experiments':
    stability_view();footer();st.stop()
if view=='data':
    data_view();footer();st.stop()
if view=='pipeline':
    pipeline_view();footer();st.stop()
mode=st.radio('Changes',['charts','ten'],format_func=lambda k:tx('mode_'+k),horizontal=True,label_visibility='collapsed',key='seg_mode') if view=='changes' else None

hero_top=None
if profile_pid:
    country=profile_pid.split(':',1)[0]
    if country not in COUNTRIES:country=list(COUNTRIES)[0]
    backcol,_=st.columns([1,3])
    with backcol:
        if st.button('← '+tx('back'),key='back_map',type='primary',width='stretch'):st.session_state.pop('profile_page',None);st.query_params.clear();st.rerun()
else:
    countrycol,variantcol=st.columns([1.25,3],vertical_alignment='bottom')
    with countrycol:
        with st.container(key='country_row'):country=st.radio(tx('country'),list(COUNTRIES),format_func=lambda k:COUNTRIES[k]['label'],horizontal=True,key='country')
cfg=COUNTRIES[country];geo=load_geo(country)
regions=[f['properties']['id'] for f in geo['features']]
region_key='region_'+country
if 'pending_region' in st.session_state:st.session_state[region_key]=st.session_state.pop('pending_region')
if st.session_state.get(region_key) not in regions:st.session_state[region_key]=cfg['default']
run_ids=[k for k,v in RUNS.items() if v['country']==country]
def _default_run():
    local=LOCAL_LANGUAGE.get(country,'en')
    for arm in ('demographic','cultural','persona'):
        for i,k in enumerate(run_ids):
            if RUNS[k]['arm']==arm and RUNS[k]['language']==local and k not in UNLINKED:return i
    return 0
default_b=_default_run()
metric='probability'
if view=='election':
    if profile_pid:
        hero=st.container(key='profile_hero');hero_top=hero.container()
        with hero:
            with st.container(key='variant_row'):run_b=st.radio(tx('variant'),run_ids,index=default_b,format_func=run_label,horizontal=True,key='variant_'+country,label_visibility='collapsed')
    else:
        with variantcol:
            with st.container(key='variant_row'):run_b=st.radio(tx('variant'),run_ids,index=default_b,format_func=run_label,horizontal=True,key='variant_'+country)
    run_a=baseline(run_b) if baseline(run_b) not in UNLINKED else run_b
elif mode=='ten':
    first_arm='cultural' if country=='el_salvador' else 'demographic'
    after_ids=[r for r in run_ids if RUNS[r]['arm'] not in ('demographic',first_arm)]
    with variantcol:
        with st.container(key='variant_row'):run_b=st.radio(tx('variant'),after_ids,format_func=run_label,horizontal=True,key='ten_'+country)
    run_a=next(k for k in run_ids if RUNS[k]['arm']==first_arm and RUNS[k]['language']==RUNS[run_b]['language'])
else:
    linked=[i for i,k in enumerate(run_ids) if k not in UNLINKED]
    default_a=linked[0];default_ab=next((i for i in linked if i!=default_a and RUNS[run_ids[i]]['arm']!='demographic'),default_b)
    x,y=st.columns(2)
    with x:
        with st.container(key='variant_row_a'):run_a=st.radio('A · '+tx('baseline'),run_ids,index=default_a,format_func=run_label,horizontal=True,key='run_a_'+country)
    with y:
        with st.container(key='variant_row_b'):run_b=st.radio('B · '+tx('scenario'),run_ids,index=default_ab,format_func=run_label,horizontal=True,key='run_b_'+country)
    metric='choice'  # vote share: each persona's top choice

if ({run_b} if view=='election' else {run_a,run_b})&UNLINKED:st.warning(tx('unlinked_map'))
if full_profile_page():
    footer();st.stop()

if view=='election':
    b=load_run(run_b,version)
    summary_b,national_b=load_summary(run_b,country,metric,version)
    if not st.session_state.get('map_focus_'+country):national_board()
    elif country=='usa':electoral_summary(summary_b,run_label(run_b))
    left,right=st.columns([2.8,1],gap='medium')
    with left:
        map_view(summary_b)
    with right:
        with st.container(key='persona_panel'):
            if run_b in UNLINKED:
                st.caption(tx('unlinked_map'))
            elif st.session_state.get('map_focus_'+country):
                persona_list(b)
            elif country=='usa':
                example_states(b)
            else:
                st.markdown('<div class="territory-intro"><h3>'+esc(tx('choose_territory'))+'</h3><p>'+esc(tx('territory_intro'))+'</p><p>'+esc(tx('persona_intro'))+'</p></div>',unsafe_allow_html=True)
                # Real deterministic sample identities, not decorative placeholder voters.
                st.markdown('<div class="preview-robots">'+''.join(f'<img width="38" height="38" alt="{esc(pid)}" src="{robot(pid)}">' for pid in b.pid.iloc[:3])+'</div>',unsafe_allow_html=True)
                st.caption(tx('agents'))
    st.markdown(f'<div class="how-card"><span class="pill">{esc(cfg["label"])}</span><h3>{esc(tx("how_title"))}</h3><p>{esc(tx("how_"+country))}</p></div>',unsafe_allow_html=True)
    if country=='usa':electoral_method()

elif mode=='ten':
    field=ADDED_FIELD.get(RUNS[run_b]['arm'])
    st.markdown(f'<div class="prompt-box"><b>{esc(g("prompt_changed"))}</b>A · {esc(run_label(run_a))} → B · {esc(run_label(run_b))}: <span class="add">B {esc(g("adds"))} {esc((field or "").replace("_"," "))}</span>.</div>',unsafe_allow_html=True)
    page_key='ten_page_'+run_b
    with st.container(key='ten_dice'):
        if st.button(tx('other_people'),icon=':material/casino:',key='ten_shuffle'):st.session_state[page_key]=st.session_state.get(page_key,0)+1;st.rerun()
    ten_table(sample_ids((run_a,run_b),version,page=st.session_state.get(page_key,0)))

else:
    a=load_run(run_a,version)
    pair=load_pair(run_a,run_b,version)
    pa,pb=pair_frames(pair)
    sa,na=aggregates(pa,country,metric);sb,nb=aggregates(pb,country,metric)
    n=len(cfg['candidates'])
    st.markdown(f'#### {esc(g("national"))} · A → B')
    strip=''.join(f'<div><small style="color:{ink(cfg["colors"][i])}">{esc(c)}</small><span class="from">{na[f"p{i}"]:.1f}%</span>→ <span class="v" style="color:{ink(cfg["colors"][i])}">{nb[f"p{i}"]:.1f}%</span><span class="d">{nb[f"p{i}"]-na[f"p{i}"]:+.1f} pp</span><span class="sub">{esc(tx(metric))}</span></div>' for i,c in enumerate(cfg['candidates']))
    cells=n+1
    if country=='usa':
        ea,eb=electoral_college(aggregates(pa,country,'probability')[0]),electoral_college(aggregates(pb,country,'probability')[0]);cells+=1
        w=eb['winner'];lead=tx('sim_winner')+': '+cfg['candidates'][w] if w is not None else no_winner(eb['totals'])
        strip+=f'<div><small>{esc(tx("ec_label"))}</small><span class="from">{ea["totals"][0]}–{ea["totals"][1]}</span>→ <span class="v">{eb["totals"][0]}–{eb["totals"][1]}</span><span class="sub">{esc(lead)}{esc(unassigned_note(eb["unassigned"]))}</span></div>'
    strip+=f'<div><small>{esc(tx("changed"))}</small><span class="v">{100*pair.winner_a.ne(pair.winner_b).mean():.1f}%</span><span class="sub">{len(pair):,} {esc(tx("matched").lower())}</span></div>'
    st.markdown(f'<div class="national-strip" style="--n:{cells}">{strip}</div>',unsafe_allow_html=True)
    map_view(sa,sb,mode='compare',key='comparison',zoom=False)
    region=st.session_state[region_key]
    focused=False
    left,right=st.columns([1.4,1])
    with left:
        st.subheader(tx('flows'))
        labels=cfg['candidates']+[tx('tie')];colors=cfg['colors']+['#8a9ba5']
        filtered=pair.loc[pair.region_a.eq(region)] if focused else pair
        flows=filtered.groupby(['winner_a','winner_b']).size().reset_index(name='count')
        def node(v):return n if v<0 else int(v)
        def flow_color(v):
            color=colors[node(v)].lstrip('#')
            return 'rgba('+','.join(str(int(color[i:i+2],16)) for i in (0,2,4))+',0.28)'
        fig=go.Figure(go.Sankey(arrangement='snap',node=dict(label=['A · '+c for c in labels]+['B · '+c for c in labels],color=colors+colors,pad=25,thickness=18,line=dict(color='#ffffff',width=1)),link=dict(source=[node(v) for v in flows.winner_a],target=[n+1+node(v) for v in flows.winner_b],value=flows['count'],color=[flow_color(v) for v in flows.winner_a],hovertemplate='%{source.label} → %{target.label}<br>%{value:,} '+tx('agents')+'<extra></extra>')))
        fig.update_layout(height=380,margin=dict(l=24,r=24,t=28,b=28),paper_bgcolor='#ffffff',plot_bgcolor='#ffffff',font=dict(size=14,color='#182b3b'))
        st.plotly_chart(fig,width='stretch',key='sankey',config={'displayModeBar':False})
    with right:
        scope=nice_region(region) if focused else tx('national')
        st.subheader('A → B · '+scope)
        if focused and region in sa.index and sa.loc[region,'valid_n']>0:
            values_a=[sa.loc[region,f'p{i}'] for i in range(n)];values_b=[sb.loc[region,f'p{i}'] for i in range(n)]
        else:
            values_a=[na[f'p{i}'] for i in range(n)];values_b=[nb[f'p{i}'] for i in range(n)]
        table=pd.DataFrame({tx('candidate'):cfg['candidates'],'A':values_a,'B':values_b})
        table['Δ pp']=table.B-table.A;st.dataframe(table.round(1),hide_index=True,width='stretch')
        if country!='brazil':
            with st.expander('2024 · Observed election benchmark'):
                if country=='usa':
                    real=[tu.NATIONAL_2024['Trump']['popular_vote_pct'],tu.NATIONAL_2024['Harris']['popular_vote_pct']] if not focused else [tu.ELECTION_2024[region][1],tu.ELECTION_2024[region][2]]
                else:
                    real=list(tu.sv_real_shares(by_department=focused).loc[region if focused else 'El Salvador'])
                total=sum(real)
                st.dataframe(pd.DataFrame({tx('candidate'):cfg['candidates'],'Observed: all valid votes (%)':real,'Observed: simulated candidates only (%)':[v/total*100 for v in real]}).round(1),hide_index=True,width='stretch')
                st.caption('Source: election reference tables documented in THESIS_LOG.md and thesis_utils.py. Other ballot candidates are outside this simulation. These observed vote shares are not calibrated model probabilities.')
    with st.expander(tx('more_analyses')):
        ci=st.radio(tx('candidate'),range(n),format_func=lambda i:cfg['candidates'][i],horizontal=True,key='seg_delta_candidate')
        st.markdown('**'+tx('difference')+'**')
        diff={r:dict(delta=float(sb.loc[r,f'p{ci}']-sa.loc[r,f'p{ci}']),n=int(sb.loc[r,'valid_n'])) for r in sa.index.intersection(sb.index) if sa.loc[r,'valid_n']>0 and sb.loc[r,'valid_n']>0}
        map_view(sa,sb,'difference','delta_map',diff,zoom=False)
        st.markdown('**'+tx('distribution')+'**')
        fig=go.Figure()
        for label,side,color in [('A','a','#7c93a1'),('B','b','#176957')]:
            values=np.sort(filtered[[f'p{i}_{side}' for i in range(n)]].to_numpy(),axis=1)
            fig.add_trace(go.Histogram(x=values[:,-1]-values[:,-2],nbinsx=20,name=label,marker_color=color,opacity=.65))
        fig.update_layout(barmode='overlay',height=260,xaxis_title=tx('margin'),yaxis_title=tx('agents'),margin=dict(l=0,r=10,t=10,b=35),paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='#fff')
        st.plotly_chart(fig,width='stretch',key='margins')

footer()
