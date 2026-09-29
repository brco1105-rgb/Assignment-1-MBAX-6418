"""Generate the final offline Step 7 dashboard from saved balanced results.

Usage:
  python scripts/generate_final_dashboard.py

Inputs are the frozen balanced sample and saved Step 6B/6C outputs. The script does
not make model calls or access the network.
"""
from __future__ import annotations

import csv
import html
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "balanced_sample.csv"
PREDICTIONS = ROOT / "results" / "balanced_raw_results.csv"
OUT = ROOT / "final_dashboard.html"
LABELS = ("POSITIVE", "NEUTRAL", "NEGATIVE")


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def esc(value):
    return html.escape(value or "")


def main():
    sample = read_csv(SAMPLE)
    pred = read_csv(PREDICTIONS)
    if len(sample) != 150 or len(pred) != 150:
        raise ValueError("Expected exactly 150 balanced reviews and 150 saved result rows")

    by_id = {str(r["sample_id"]): r for r in sample}
    rows = []
    matrix = {a: {p: 0 for p in (*LABELS, "NO_RESPONSE")} for a in LABELS}
    actual_counts = Counter()
    pred_counts = Counter()
    star_counts = Counter()
    correct = valid = 0

    for r in pred:
        rid = str(r.get("review_number") or r.get("sample_id"))
        s = by_id[rid]
        actual = r.get("correct_sentiment") or s["correct_sentiment"]
        predicted = (r.get("new_sentiment") or "").strip() or "NO_RESPONSE"
        format_valid = str(r.get("format_valid", "")).lower() == "true"
        is_correct = str(r.get("is_correct", "")).lower() == "true"
        valid += int(format_valid)
        correct += int(is_correct)
        actual_counts[actual] += 1
        pred_counts[predicted] += 1
        star_counts[int(float(s["star_rating"]))] += 1
        matrix[actual][predicted] += 1
        rows.append({**s, **r, "actual": actual, "predicted": predicted, "is_correct_bool": is_correct})

    if dict(actual_counts) != {"POSITIVE": 50, "NEUTRAL": 50, "NEGATIVE": 50}:
        raise ValueError(f"Unexpected actual class counts: {dict(actual_counts)}")

    recall = {lab: matrix[lab][lab] / actual_counts[lab] for lab in LABELS}
    accuracy = correct / len(rows)
    balanced_accuracy = sum(recall.values()) / 3

    def pct(x): return f"{x*100:.2f}%"
    max_star = max(star_counts.values())
    max_pred = max(pred_counts.values())

    star_bars = "".join(
        f'<div class="bar-row"><span>{star} star</span><div class="bar-track"><div class="bar-fill" style="width:{star_counts[star]/max_star*100:.2f}%"></div></div><strong>{star_counts[star]}</strong></div>'
        for star in range(1, 6)
    )

    avp = "".join(
        f'<div class="compare-row"><span>{lab.title()}</span><div class="double-bars"><div class="mini actual" style="width:100%"></div><div class="mini predicted" style="width:{pred_counts[lab]/max_pred*100:.2f}%"></div></div><span class="nums">Actual 50 · Pred {pred_counts[lab]}</span></div>'
        for lab in LABELS
    ) + f'<div class="compare-row"><span>No response</span><div class="double-bars"><div class="mini predicted" style="width:{pred_counts["NO_RESPONSE"]/max_pred*100:.2f}%"></div></div><span class="nums">{pred_counts["NO_RESPONSE"]}</span></div>'

    matrix_rows = ""
    for actual in LABELS:
        cells = ""
        for predicted in (*LABELS, "NO_RESPONSE"):
            cls = "hit" if actual == predicted else ("missing" if predicted == "NO_RESPONSE" else "miss")
            cells += f'<td class="{cls}"><strong>{matrix[actual][predicted]}</strong><span>{predicted.replace("_", " ").title()}</span></td>'
        matrix_rows += f'<tr><th>{actual}</th>{cells}</tr>'

    recall_cards = "".join(
        f'<div class="recall-card"><div><span>{lab}</span><strong>{pct(recall[lab])}</strong></div><div class="bar-track"><div class="bar-fill {"neg" if lab=="NEGATIVE" else ("neu" if lab=="NEUTRAL" else "")}" style="width:{recall[lab]*100:.2f}%"></div></div><small>{matrix[lab][lab]} of 50 correct</small></div>'
        for lab in LABELS
    )

    table_rows = ""
    for r in rows:
        status = "Correct" if r["is_correct_bool"] else ("No response" if r["predicted"] == "NO_RESPONSE" else "Incorrect")
        cls = "good" if status == "Correct" else ("warn" if status == "No response" else "bad")
        table_rows += f'''<tr data-status="{status.lower().replace(' ','-')}" data-actual="{r['actual']}" data-pred="{r['predicted']}">
<td>{r['sample_id']}</td><td><details><summary>{esc(r.get('title'))}</summary><div class="review-text">{esc(r.get('text'))}</div></details></td>
<td>{int(float(r['star_rating']))}</td><td><span class="tag {r['actual'].lower()}">{r['actual']}</span></td>
<td><span class="tag {r['predicted'].lower()}">{r['predicted']}</span></td><td><span class="{cls}">{status}</span></td></tr>'''

    template = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Gift Cards — Balanced Three-Class Sentiment Evaluation</title>
<style>
:root{{--paper:#f4f5f1;--surface:#fff;--ink:#203334;--muted:#667675;--line:#dce5df;--navy:#143a3c;--green:#167568;--green-soft:#e8f4ef;--amber:#9b6a1f;--amber-soft:#fff4dd;--red:#a04430;--red-soft:#fff0e9;--blue:#536f86;--blue-soft:#eef3f7;--radius:14px;--font:'Segoe UI','Helvetica Neue',Arial,sans-serif}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.5 var(--font)}}.shell{{max-width:1260px;margin:auto;padding:0 36px}}.topbar{{background:var(--navy);color:white}}.topbar .shell{{display:flex;justify-content:space-between;align-items:center;min-height:58px}}.wordmark{{font-weight:800;letter-spacing:.14em;font-size:13px}}.topnote{{font-size:12px;color:#d6e7e3}}header{{padding-top:30px}}h1{{font-size:32px;letter-spacing:-.03em;margin:0 0 8px}}.subtitle{{color:var(--muted);max-width:850px}}.metrics{{display:grid;grid-template-columns:1.6fr repeat(4,1fr);gap:12px;margin:26px 0 20px}}.metric{{background:var(--surface);border:1px solid var(--line);border-radius:var(--radius);padding:20px}}.metric.primary{{background:var(--navy);color:#fff;border-color:var(--navy)}}.metric .label{{font-size:12px;font-weight:700}}.metric .value{{font-size:34px;font-weight:800;margin:8px 0 4px}}.metric small{{color:var(--muted)}}.primary small{{color:#d6e7e3}}.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin:18px 0}}.panel{{background:var(--surface);border:1px solid var(--line);border-radius:var(--radius);padding:24px}}.panel h2{{font-size:19px;margin:0 0 6px}}.caption{{color:var(--muted);font-size:12px;margin:0 0 18px}}.bar-row{{display:grid;grid-template-columns:70px 1fr 42px;gap:10px;align-items:center;margin:11px 0}}.bar-track{{height:10px;background:#e8ece9;border-radius:999px;overflow:hidden}}.bar-fill{{height:100%;background:var(--green)}}.bar-fill.neg{{background:var(--red)}}.bar-fill.neu{{background:var(--amber)}}.compare-row{{display:grid;grid-template-columns:90px 1fr 145px;gap:12px;align-items:center;margin:14px 0}}.double-bars{{display:grid;gap:5px}}.mini{{height:8px;border-radius:999px}}.mini.actual{{background:var(--blue)}}.mini.predicted{{background:var(--green)}}.nums{{font-size:12px;color:var(--muted);text-align:right}}.matrix{{width:100%;border-collapse:separate;border-spacing:6px}}.matrix th{{font-size:11px;text-align:left;color:var(--muted)}}.matrix td{{text-align:center;padding:14px 8px;border-radius:9px;background:#f2f5f3}}.matrix td.hit{{background:var(--green-soft);color:var(--green)}}.matrix td.miss{{background:var(--red-soft);color:var(--red)}}.matrix td.missing{{background:var(--blue-soft);color:var(--blue)}}.matrix td strong{{display:block;font-size:26px}}.matrix td span{{font-size:10px;text-transform:uppercase}}.recall-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}}.recall-card{{border:1px solid var(--line);border-radius:12px;padding:17px}}.recall-card>div:first-child{{display:flex;justify-content:space-between;align-items:baseline}}.recall-card strong{{font-size:26px}}.recall-card small{{display:block;color:var(--muted);margin-top:7px}}.callout{{background:var(--amber-soft);border:1px solid #ead8ae;border-radius:var(--radius);padding:20px;margin:18px 0}}.callout strong{{font-size:30px;color:var(--amber)}}.filters{{display:flex;flex-wrap:wrap;gap:8px;margin:14px 0}}button{{font:600 12px var(--font);padding:9px 12px;border:1px solid var(--line);border-radius:7px;background:white;cursor:pointer}}button.active{{background:var(--navy);color:white;border-color:var(--navy)}}#count{{margin-left:auto;align-self:center;color:var(--muted);font-size:12px}}.table-wrap{{background:white;border:1px solid var(--line);border-radius:var(--radius);overflow:hidden}}.table-scroll{{max-height:680px;overflow:auto}}table.reviews{{width:100%;border-collapse:collapse;min-width:850px}}.reviews th{{position:sticky;top:0;background:#fafbf9;text-align:left;font-size:10px;text-transform:uppercase;color:var(--muted);padding:12px;border-bottom:1px solid var(--line)}}.reviews td{{padding:12px;border-bottom:1px solid #edf0ee;vertical-align:top;font-size:12px}}details summary{{cursor:pointer;font-weight:650}}.review-text{{white-space:pre-wrap;margin-top:10px;color:#415252}}.tag{{display:inline-block;padding:4px 7px;border-radius:5px;font-size:10px;font-weight:800}}.tag.positive{{background:var(--green-soft);color:var(--green)}}.tag.neutral{{background:var(--amber-soft);color:var(--amber)}}.tag.negative{{background:var(--red-soft);color:var(--red)}}.tag.no_response{{background:var(--blue-soft);color:var(--blue)}}.good{{color:var(--green);font-weight:700}}.bad{{color:var(--red);font-weight:700}}.warn{{color:var(--blue);font-weight:700}}footer{{color:var(--muted);font-size:11px;padding:26px 0 40px}}@media(max-width:850px){{.metrics{{grid-template-columns:1fr 1fr}}.metric.primary{{grid-column:1/-1}}.grid2,.recall-grid{{grid-template-columns:1fr}}.shell{{padding:0 18px}}}}
</style></head><body><div class="topbar"><div class="shell"><span class="wordmark">REVIEW / LAB</span><span class="topnote">AMAZON REVIEWS 2023 · GIFT CARDS</span></div></div><header class="shell"><h1>Balanced three-class sentiment evaluation</h1><p class="subtitle">Fixed 150-review balanced sample. Benchmark: 4–5 stars = POSITIVE, 3 stars = NEUTRAL, 1–2 stars = NEGATIVE. Model input: title and text only.</p></header><main class="shell">
<section class="metrics"><div class="metric primary"><div class="label">Overall benchmark agreement</div><div class="value">{pct(accuracy)}</div><small>{correct} correct out of 150; the one missing response is counted as not correct.</small></div><div class="metric"><div class="label">Valid responses</div><div class="value">{valid}</div><small>of 150 reviews</small></div><div class="metric"><div class="label">Balanced accuracy</div><div class="value">{pct(balanced_accuracy)}</div><small>equal class weighting</small></div><div class="metric"><div class="label">Neutral recall</div><div class="value">{pct(recall['NEUTRAL'])}</div><small>9 of 50</small></div><div class="metric"><div class="label">No response</div><div class="value">{pred_counts['NO_RESPONSE']}</div><small>preserved, not rerun</small></div></section>
<div class="grid2"><section class="panel"><h2>Star-rating distribution</h2><p class="caption">Balanced by three sentiment classes, not by individual star values.</p>{star_bars}</section><section class="panel"><h2>Actual vs. predicted class counts</h2><p class="caption">The model underproduced NEUTRAL and overproduced NEGATIVE.</p>{avp}</section></div>
<section class="panel"><h2>Three-class confusion matrix</h2><p class="caption">Rows are benchmark classes; columns are model predictions.</p><table class="matrix"><thead><tr><th>Actual ↓ / Predicted →</th><th>Positive</th><th>Neutral</th><th>Negative</th><th>No response</th></tr></thead><tbody>{matrix_rows}</tbody></table></section>
<section class="panel" style="margin-top:18px"><h2>Recall by benchmark class</h2><div class="recall-grid">{recall_cards}</div></section>
<section class="callout"><strong>32 of 50</strong><p>benchmark-NEUTRAL reviews were classified as NEGATIVE. Only 9 of 50 were classified as NEUTRAL; another 9 were classified as POSITIVE.</p></section>
<section class="panel"><h2>How to read this against the first run</h2><p>The first 100-review binary run reached <strong>98.00%</strong> agreement on a batch that was <strong>93%</strong> positive. This balanced run uses equal benchmark classes and adds NEUTRAL, so the lower three-class score cannot be attributed to balancing alone.</p></section>
<h2>Review-level results</h2><div class="filters"><button class="active" data-filter="all">All</button><button data-filter="correct">Correct</button><button data-filter="incorrect">Incorrect</button><button data-filter="neutral-errors">Neutral errors</button><button data-filter="no-response">No response</button><span id="count">150 of 150 visible</span></div><div class="table-wrap"><div class="table-scroll"><table class="reviews"><thead><tr><th>ID</th><th>Review</th><th>Stars</th><th>Benchmark</th><th>Prediction</th><th>Status</th></tr></thead><tbody>{table_rows}</tbody></table></div></div>
<p style="font-size:12px;color:var(--muted)">Generated entirely from saved local outputs; no model calls are made by this dashboard generator.</p></main><footer class="shell">MBAX 6418 · Assignment 1</footer><script>const buttons=[...document.querySelectorAll('[data-filter]')],rows=[...document.querySelectorAll('table.reviews tbody tr')],count=document.getElementById('count');function applyFilter(kind){{let visible=0;rows.forEach(row=>{{const status=row.dataset.status,actual=row.dataset.actual,pred=row.dataset.pred;let show=true;if(kind==='correct')show=status==='correct';else if(kind==='incorrect')show=status==='incorrect';else if(kind==='neutral-errors')show=actual==='NEUTRAL'&&pred!=='NEUTRAL';else if(kind==='no-response')show=status==='no-response';row.hidden=!show;if(show)visible++}});count.textContent=`${{visible}} of ${{rows.length}} visible`;buttons.forEach(b=>b.classList.toggle('active',b.dataset.filter===kind))}}buttons.forEach(b=>b.addEventListener('click',()=>applyFilter(b.dataset.filter)));</script></body></html>'''

    OUT.write_text(template, encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
