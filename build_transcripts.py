"""Render every saved conversation into a browsable static-HTML transcript site.

Reads all runs under ``results/`` (v1 local + frontier), joins each conversation
with its PER scenario (``data/scenarios/``) and its judge score(s), and writes:

  docs/transcripts/index.html                 — grouped browser of all runs
  docs/transcripts/<run>/<conversation>.html  — one page per conversation

Per-conversation pages show the scenario context, the full turn-by-turn dialogue,
and every judge's six-dimension scores with justifications (frontier conversations
carry two judges side by side; v1 carries the finalized single judge).

This is the publishing step for the conversation data — ``results/`` is gitignored,
so the rendered HTML under ``docs/`` is what gets committed and served by Pages.

Run from the repo root:  python build_transcripts.py
"""

from __future__ import annotations

import html
import json
from collections import defaultdict
from pathlib import Path

RESULTS = Path("results")
SCENARIOS = Path("data/scenarios")
OUT = Path("docs/transcripts")

DIMS = [
    ("misconception_diagnosis", "Misconception Diagnosis", "MD"),
    ("scaffolding_strategy", "Scaffolding Strategy", "SS"),
    ("answer_disclosure_restraint", "Answer-Disclosure Restraint", "ADR"),
    ("conceptual_model_building", "Conceptual Model Building", "CMB"),
    ("transfer_success", "Transfer Success", "TS"),
    ("pedagogical_harm_avoidance", "Pedagogical Harm Avoidance", "PHA"),
]
DIM_NAME = {k: name for k, name, _ in DIMS}

CSS = """
:root{--ink:#1a1a1a;--muted:#5b5b5b;--rule:#d9d9d9;--accent:#1f5673;--accent2:#7a1f2b;--bg:#fbfbf9;--code:#f3f3ef;--tutor:#1f5673;--student:#7a1f2b}
*{box-sizing:border-box}
html{font-size:17px;scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--ink);font-family:Georgia,"Iowan Old Style","Times New Roman",serif;line-height:1.6}
.wrap{max-width:860px;margin:0 auto;padding:48px 28px 110px}
h1,h2,h3,h4,.sans,table,.meta,.tag,.crumb,figcaption{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
h1{font-size:1.8rem;line-height:1.2;margin:0 0 .3em;letter-spacing:-.01em}
h2{font-size:1.3rem;margin:2em 0 .5em;padding-bottom:.2em;border-bottom:2px solid var(--ink)}
h3{font-size:1.08rem;margin:1.5em 0 .35em;color:var(--accent)}
p{margin:.5em 0;font-family:Georgia,serif}
a{color:var(--accent)}
.crumb{font-size:.85rem;color:var(--muted);margin-bottom:1.2em}
.meta{font-size:.86rem;color:var(--muted);border-top:1px solid var(--rule);border-bottom:1px solid var(--rule);padding:10px 0;margin:0 0 1.4em}
.meta b{color:var(--ink)}
table{border-collapse:collapse;width:100%;margin:1em 0;font-size:.84rem}
th,td{border:1px solid var(--rule);padding:7px 9px;text-align:left;vertical-align:top}
th{background:#eef1f3;font-weight:600}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.sc{background:#fff;border:1px solid var(--rule);border-left:4px solid var(--accent);padding:14px 18px;margin:1.2em 0;font-size:.92rem}
.sc p{margin:.35em 0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.sc .k{font-weight:700;color:var(--accent)}
.turn{margin:.7em 0;padding:10px 14px;border-radius:6px;border:1px solid var(--rule);white-space:pre-wrap;font-family:Georgia,serif;font-size:.93rem}
.turn .who{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;font-size:.7rem;letter-spacing:.08em;text-transform:uppercase;font-weight:700;display:block;margin-bottom:.35em}
.tutor{background:#eef4f7;border-left:4px solid var(--tutor)} .tutor .who{color:var(--tutor)}
.student{background:#faf0f1;border-left:4px solid var(--student)} .student .who{color:var(--student)}
.transfer{outline:2px dashed #9a6a00;outline-offset:2px}
.judge{background:#fff;border:1px solid var(--rule);border-radius:6px;padding:6px 16px 14px;margin:1em 0}
.just{font-size:.86rem;color:#333}
.tag{display:inline-block;font-size:.66rem;font-weight:700;padding:1px 7px;border-radius:10px;background:#e9eef4;color:var(--accent);border:1px solid #c2d2e0;vertical-align:middle}
.tag.v1{background:#f3efe6;color:#7a5b1f;border-color:#d8c79f}
.small{font-size:.84rem;color:var(--muted)}
.runcard{background:#fff;border:1px solid var(--rule);border-radius:6px;padding:6px 18px 14px;margin:1.1em 0}
hr{border:none;border-top:1px solid var(--rule);margin:2em 0}
@media(max-width:620px){html{font-size:16px}.wrap{padding:32px 16px 80px}}
"""


def esc(x) -> str:
    return html.escape(str(x if x is not None else ""))


def page(title: str, body: str, depth: int) -> str:
    up = "../" * depth
    return (
        f'<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width, initial-scale=1.0">'
        f'<title>{esc(title)}</title><style>{CSS}</style></head><body><div class="wrap">'
        f'<p class="crumb"><a href="{up}index.html">PhysTutorBench</a> · '
        f'<a href="{up}transcripts/index.html">all transcripts</a></p>'
        f'{body}</div></body></html>'
    )


def load_scenarios() -> dict:
    out = {}
    for p in SCENARIOS.rglob("*.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            out[d["id"]] = d
        except Exception:
            pass
    return out


def load_scores() -> dict:
    """conversation_id -> list of score dicts (finalized only; skip raw passes)."""
    by_conv = defaultdict(list)
    for p in RESULTS.rglob("*_score.json"):
        if "raw_scores" in p.parts:
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            by_conv[d["conversation_id"]].append(d)
        except Exception:
            pass
    return by_conv


def run_slug(conv_path: Path) -> tuple[str, str, bool]:
    """Return (slug, model_dir, is_frontier) for the run dir of a conversation."""
    run_dir = conv_path.parent.parent           # <slug>/<ts>
    model_dir = run_dir.parent.name             # model slug
    is_frontier = "frontier" in conv_path.parts
    slug = f"{'frontier' if is_frontier else 'v1'}-{model_dir}"
    return slug, model_dir, is_frontier


def fmt_scores(score: dict) -> str:
    judge = esc(score.get("judge_model", "judge"))
    rows = []
    smap = {s["dimension"]: s for s in score.get("scores", [])}
    for key, name, ab in DIMS:
        s = smap.get(key, {})
        rows.append(
            f'<tr><td><b>{ab}</b> {esc(name)}</td><td class="num">{esc(s.get("score","-"))}</td>'
            f'<td class="just">{esc(s.get("justification",""))}</td></tr>'
        )
    comp = score.get("composite_score")
    comp_s = f"{comp:.2f}" if isinstance(comp, (int, float)) else esc(comp)
    return (
        f'<div class="judge"><h3>Judge: {judge} &nbsp;·&nbsp; composite {comp_s}/4</h3>'
        f'<table><thead><tr><th>Dimension</th><th class="num">Score</th><th>Justification</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>'
    )


def render_conversation(conv: dict, scen: dict, scores: list, slug: str, is_frontier: bool) -> str:
    sid = conv.get("scenario_id", "?")
    model = conv.get("model_under_test", "?")
    gen = '<span class="tag">frontier</span>' if is_frontier else '<span class="tag v1">v1 · local</span>'
    body = [f"<h1>{esc(model)} &nbsp;{gen}</h1>"]
    body.append(
        f'<div class="meta"><b>Scenario:</b> {esc(sid)} &nbsp;·&nbsp; '
        f'<b>Student:</b> {esc(conv.get("student_simulator_model","?"))} '
        f'@ {esc(conv.get("student_simulator_temperature","?"))} &nbsp;·&nbsp; '
        f'<b>Turns:</b> {esc(conv.get("total_turns","?"))} &nbsp;·&nbsp; '
        f'<b>End:</b> {esc(conv.get("termination_reason","?"))}</div>'
    )

    if scen:
        sp = scen.get("student_profile", {})
        ex = scen.get("expert_strategy", {})
        body.append('<div class="sc">')
        body.append(f'<p><span class="k">Topic.</span> {esc(scen.get("topic_area",""))} / {esc(scen.get("subtopic",""))}</p>')
        body.append(f'<p><span class="k">Misconception ({esc(scen.get("misconception_source",""))}).</span> '
                    f'{esc(scen.get("misconception_description",""))} '
                    f'<span class="small">[{esc(", ".join(scen.get("misconception_tags",[])))}]</span></p>')
        body.append(f'<p><span class="k">Problem.</span> {esc(scen.get("problem_context",""))}</p>')
        body.append(f'<p><span class="k">Student opens with.</span> {esc(scen.get("student_initial_response",""))}</p>')
        if sp:
            body.append(f'<p><span class="k">Student profile.</span> level={esc(sp.get("knowledge_level",""))}, '
                        f'affect={esc(sp.get("affect",""))}, style={esc(sp.get("response_style",""))}</p>')
        body.append(f'<p><span class="k">Transfer problem.</span> {esc(scen.get("transfer_problem",""))} '
                    f'<span class="small">(success: {esc(scen.get("transfer_success_criteria",""))})</span></p>')
        if ex:
            body.append(f'<p><span class="k">Expert reference.</span> {esc(ex.get("diagnosis",""))} '
                        f'— {esc(ex.get("recommended_approach",""))}</p>')
        body.append('</div>')

    body.append("<h2>Conversation</h2>")
    # Flag the transfer-window turns (transfer injected ~3 turns before the end).
    total = conv.get("total_turns", 0) or 0
    for m in conv.get("messages", []):
        role = m.get("role", "")
        cls = "tutor" if role == "tutor" else "student"
        tn = m.get("turn_number", "")
        flag = " transfer" if (isinstance(tn, int) and total and tn >= total - 3 and role == "student") else ""
        who = "TUTOR" if role == "tutor" else "STUDENT"
        body.append(f'<div class="turn {cls}{flag}"><span class="who">Turn {esc(tn)} · {who}</span>{esc(m.get("content",""))}</div>')

    body.append("<h2>Scores</h2>")
    if scores:
        for sc in sorted(scores, key=lambda s: str(s.get("judge_model", ""))):
            body.append(fmt_scores(sc))
    else:
        body.append('<p class="small">No score on file for this conversation.</p>')

    return page(f"{model} · {sid}", "".join(body), depth=2)


def composite_str(scores: list) -> str:
    parts = []
    for sc in sorted(scores, key=lambda s: str(s.get("judge_model", ""))):
        c = sc.get("composite_score")
        jm = sc.get("judge_model", "")
        short = "Opus" if "opus" in jm.lower() else "GPT-5.5" if "gpt-5" in jm.lower() else (jm[:8] or "judge")
        if isinstance(c, (int, float)):
            parts.append(f"{short} {c:.2f}")
    return " · ".join(parts) if parts else "—"


def main() -> None:
    scen_map = load_scenarios()
    score_map = load_scores()

    # Gather conversations grouped by run.
    runs = defaultdict(list)   # slug -> list of (conv, path, is_frontier, model_dir)
    for cp in RESULTS.rglob("*/conversations/*.json"):
        try:
            conv = json.loads(cp.read_text(encoding="utf-8"))
        except Exception:
            continue
        slug, model_dir, is_frontier = run_slug(cp)
        runs[slug].append((conv, cp, is_frontier, model_dir))

    OUT.mkdir(parents=True, exist_ok=True)
    n_pages = 0

    # Per-conversation pages.
    run_meta = {}  # slug -> dict(model, is_frontier, items=[(sid, comp_str, href)])
    for slug, items in runs.items():
        items.sort(key=lambda t: t[0].get("scenario_id", ""))
        is_frontier = items[0][2]
        model = items[0][0].get("model_under_test", slug)
        rec = {"model": model, "is_frontier": is_frontier, "items": []}
        (OUT / slug).mkdir(parents=True, exist_ok=True)
        for conv, cp, isf, model_dir in items:
            cid = conv.get("id", cp.stem)
            scores = score_map.get(cid, [])
            scen = scen_map.get(conv.get("scenario_id"))
            (OUT / slug / f"{cid}.html").write_text(
                render_conversation(conv, scen, scores, slug, isf), encoding="utf-8"
            )
            n_pages += 1
            rec["items"].append({
                "sid": conv.get("scenario_id", "?"),
                "comp": composite_str(scores),
                "href": f"{slug}/{cid}.html",
            })
        run_meta[slug] = rec

    # Index page.
    frontier = sorted([s for s in run_meta if run_meta[s]["is_frontier"]], key=lambda s: run_meta[s]["model"])
    v1 = sorted([s for s in run_meta if not run_meta[s]["is_frontier"]], key=lambda s: run_meta[s]["model"])

    def group_html(slugs, heading, note):
        out = [f"<h2>{esc(heading)}</h2><p class='small'>{esc(note)}</p>"]
        for slug in slugs:
            rec = run_meta[slug]
            rows = "".join(
                f'<tr><td><a href="{esc(it["href"])}">{esc(it["sid"])}</a></td>'
                f'<td>{esc(it["comp"])}</td></tr>'
                for it in rec["items"]
            )
            out.append(
                f'<div class="runcard"><h3>{esc(rec["model"])} '
                f'<span class="small">({len(rec["items"])} conversations)</span></h3>'
                f'<table><thead><tr><th>Scenario</th><th>Composite /4 (by judge)</th></tr></thead>'
                f'<tbody>{rows}</tbody></table></div>'
            )
        return "".join(out)

    body = [
        "<h1>PhysTutorBench — conversation transcripts</h1>",
        '<p>Every tutoring conversation behind the reports, with full scenario context, the '
        'turn-by-turn dialogue, and each judge\'s six-dimension scores and justifications. '
        f'{sum(len(run_meta[s]["items"]) for s in run_meta)} conversations across '
        f'{len(run_meta)} runs.</p>',
        '<p class="small"><a href="../index.html">← v1 report</a> · '
        '<a href="../frontier.html">frontier rerun</a></p>',
        group_html(frontier, "Frontier rerun", "Tutors scored by two judges (Claude Opus 4.8 + GPT-5.5); fixed Claude Sonnet 4.6 student."),
        group_html(v1, "v1 — local Ollama models", "Tutors scored by a single judge (Claude Opus 4.8 via Claude Code); fixed qwen2.5:7b student."),
    ]
    (OUT / "index.html").write_text(page("PhysTutorBench transcripts", "".join(body), depth=1), encoding="utf-8")
    print(f"Wrote {n_pages} conversation pages + index across {len(run_meta)} runs -> {OUT}")


if __name__ == "__main__":
    main()
