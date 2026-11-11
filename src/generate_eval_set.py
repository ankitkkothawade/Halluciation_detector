"""
Build a labelled evaluation set from Spider: ask an LLM (via Groq) to write SQL
for Spider questions, then label each answer against Spider's gold SQL.

Usage (after `pip install groq`, with Spider in <project_root>/datasets/spider_data/):
    export GROQ_API_KEY=...
    python3 src/generate_eval_set.py --n 150 --out eval_set_v3.jsonl

Each output row records `finish_reason` from the API. "length" means the model
hit its token limit mid-answer; such rows are labelled None (unusable) rather
than "hallucinated" — see relabel_eval_set.py for the full label rules.

Notes:
- gpt-oss-120b is a reasoning model. Its reasoning tokens most likely share the
  --max-tokens budget (not verified; the finish_reason counts printed at the end
  of a run will show), so a small budget can leave the answer empty or cut off.
  Hence the 2048 default. (The old 300 looked too tight.)
- Spider's dev.json is grouped by database, so records are shuffled (seeded)
  before taking the first --n; otherwise a small --n covers only a few databases.
- If a model name 404s, run with --list-models to see what is live on your account.
- --degrade-schema table|column is a stress condition (see degrade_schema.py): one table or
  column that the gold query uses is left out of the schema shown to the model, so it has to
  guess a name. Labels and detectors still use the full database.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
import time
from collections import Counter
from pathlib import Path

from groq import Groq, RateLimitError

from degrade_schema import build_ddl, choose_hidden
from relabel_eval_set import label_row

DEFAULT_SPIDER_DIR = Path(__file__).resolve().parent.parent / "datasets" / "spider_data"

# Groq deprecated llama-3.3-70b-versatile on 16 Aug 2026; this is the working default.
DEFAULT_MODEL = "openai/gpt-oss-120b"

SYSTEM_PROMPT = (
    "You are a text-to-SQL assistant. Given a database schema and a "
    "natural-language question, output ONLY the SQL query that answers it. "
    "No explanation, no markdown fences, just the raw SQL."
)


def load_spider_split(spider_dir: Path, split_file: str) -> list[dict]:
    with open(spider_dir / split_file) as f:
        data = json.load(f)
    if data:
        print(f"[sanity check] keys in first record of {split_file}: {list(data[0].keys())}")
    return data


def get_create_statements(db_path: Path) -> str:
    """Real CREATE TABLE statements straight from the .sqlite file."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        cur = conn.cursor()
        cur.execute("SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL")
        return "\n".join(row[0] for row in cur.fetchall())
    finally:
        conn.close()


def clean_sql(text: str | None) -> str:
    """Strip markdown fences if the model added them despite being told not to."""
    text = (text or "").strip()
    if text.startswith("```"):
        first_line, newline, rest = text.partition("\n")
        tag = first_line[3:].strip().lower()  # "sql" in "```sql", "" in a bare "```"
        if newline and tag in {"", "sql", "sqlite"}:
            text = rest  # drop the whole fence line
        else:
            text = text[3:]  # e.g. "```SELECT 1```": there is no fence line
    text = text.rstrip()
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def create_with_backoff(client: Groq, max_retries: int = 8, **request):
    """chat.completions.create, but wait and retry when Groq answers 429 (rate limit)."""
    for attempt in range(max_retries + 1):
        try:
            return client.chat.completions.create(**request)
        except RateLimitError:
            if attempt == max_retries:
                raise
            wait = min(60, 2 ** (attempt + 1))
            print(f"[rate limit] waiting {wait}s (retry {attempt + 1}/{max_retries})", flush=True)
            time.sleep(wait)


def generate_sql(
    client: Groq,
    model: str,
    schema_ddl: str,
    question: str,
    max_tokens: int,
    reasoning_effort: str | None = None,
) -> tuple[str, str]:
    """Returns (sql, finish_reason). finish_reason == "length" means truncated."""
    kwargs = {}
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    resp = create_with_backoff(
        client,
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Schema:\n{schema_ddl}\n\nQuestion: {question}\n\nSQL:"},
        ],
        temperature=0,
        max_completion_tokens=max_tokens,
        **kwargs,
    )
    choice = resp.choices[0]
    return clean_sql(choice.message.content), choice.finish_reason


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spider-dir", type=Path, default=DEFAULT_SPIDER_DIR)
    ap.add_argument("--split-file", default="dev.json")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", type=Path, default=Path("eval_set.jsonl"))
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="max_completion_tokens per request. Reasoning models probably spend part "
        "of it on reasoning, so keep it generous (default 2048).",
    )
    ap.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high"],
        default=None,
        help="Optional. Not sent unless given, so the model's own default (medium for "
        "gpt-oss-120b) applies. Changing it changes the model's behaviour, so keep it "
        "fixed when comparing eval sets.",
    )
    ap.add_argument("--list-models", action="store_true", help="Print live Groq models and exit.")
    ap.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Keep Spider's original order (grouped by db_id) instead of shuffling first. "
        "Off by default: an unshuffled small --n covers only a handful of databases.",
    )
    ap.add_argument("--seed", type=int, default=42, help="Shuffle seed, for reproducibility.")
    ap.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip this many records (after shuffling) before taking --n. For a held-out sample "
        "that does not overlap an earlier run, keep the same --seed and set --offset to the "
        "earlier run's --n (e.g. --offset 150 after a --n 150 run).",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Continue an interrupted run: keep the rows already in --out (matched by "
        "sample_index) and generate only the missing ones. Use the same --offset, --n, "
        "--seed and --model as the interrupted run.",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace --out if it already has rows. Without --resume or --overwrite the "
        "script refuses, so an expensive partial run is not wiped by accident.",
    )
    ap.add_argument(
        "--degrade-schema",
        choices=["none", "table", "column"],
        default="none",
        help="Stress condition. 'table' / 'column' hide one table / column that the GOLD query "
        "uses from the schema shown to the model (foreign keys that touch it go too), so the "
        "model has to guess. Labels and detectors still use the full database; gold is only "
        "used to choose what to hide. The prompt schema is then written from the database's "
        "metadata instead of its CREATE statements. Use the same value with --resume.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not call the API or write --out (no key needed): print which questions would get "
        "something hidden under --degrade-schema, plus one example prompt schema, then exit.",
    )
    args = ap.parse_args()
    if args.resume and args.overwrite:
        ap.error("--resume and --overwrite cannot be combined")

    client = None if args.dry_run else Groq(api_key=os.environ["GROQ_API_KEY"])

    if args.list_models:
        for model in client.models.list().data:
            print(model.id)
        return

    all_records = load_spider_split(args.spider_dir, args.split_file)
    if not args.no_shuffle:
        random.Random(args.seed).shuffle(all_records)
    records = all_records[args.offset : args.offset + args.n]

    seen_dbs = {r["db_id"] for r in records}
    print(f"[info] sampling {len(records)} records across {len(seen_dbs)} distinct databases")

    if args.dry_run:
        n_hidden = 0
        example = None
        for i, rec in enumerate(records):
            db_path = args.spider_dir / "database" / rec["db_id"] / f"{rec['db_id']}.sqlite"
            if not db_path.exists():
                continue
            hidden = choose_hidden(args.degrade_schema, rec["query"], db_path, f"{args.seed}:{args.offset + i}")
            n_hidden += bool(hidden)
            if hidden and example is None:
                example = (rec, db_path, hidden)
            if not hidden:
                print(f"[{args.offset + i}] {rec['db_id']}: nothing hidden")
        print()
        print(f"would hide something for {n_hidden} of {len(records)} questions (--degrade-schema {args.degrade_schema})")
        if example:
            rec, db_path, hidden = example
            print()
            print(f"Example question: {rec['question']!r}")
            print(f"hidden: {hidden}")
            print("schema shown to the model:")
            print(build_ddl(db_path, hidden))
        return

    reasons: Counter = Counter()
    finishes: Counter = Counter()
    done: set[int] = set()
    n_degraded = 0
    if args.out.exists() and args.out.stat().st_size > 0:
        if args.resume:
            kept = []
            for line in open(args.out):
                try:
                    old = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a half-written last line: that row is generated again
                if "sample_index" not in old:
                    raise SystemExit("Cannot resume: these rows have no sample_index (older file format).")
                if old.get("degrade_mode", "none") != args.degrade_schema:
                    raise SystemExit(
                        f"Cannot resume: {args.out} was made with --degrade-schema "
                        f"{old.get('degrade_mode', 'none')}, not {args.degrade_schema}."
                    )
                done.add(old["sample_index"])
                n_degraded += bool(old.get("hidden"))
                reasons[old.get("label_reason")] += 1
                finishes[old.get("finish_reason")] += 1
                kept.append(line if line.endswith("\n") else line + "\n")
            args.out.write_text("".join(kept))
            print(f"[resume] keeping {len(done)} rows already in {args.out}")
        elif not args.overwrite:
            raise SystemExit(f"{args.out} already has rows. Use --resume to continue it or --overwrite to start over.")
    with open(args.out, "a" if args.resume else "w") as out_f:
        for i, rec in enumerate(records):
            if args.offset + i in done:
                continue
            db_id = rec["db_id"]
            db_path = args.spider_dir / "database" / db_id / f"{db_id}.sqlite"
            if not db_path.exists():
                print(f"[skip] {i}: no sqlite file for db_id={db_id} at {db_path}")
                continue

            hidden = choose_hidden(args.degrade_schema, rec["query"], db_path, f"{args.seed}:{args.offset + i}")
            if args.degrade_schema == "none":
                schema_ddl = get_create_statements(db_path)
            else:
                schema_ddl = build_ddl(db_path, hidden)
            generated_sql, finish_reason = generate_sql(
                client,
                args.model,
                schema_ddl,
                rec["question"],
                args.max_tokens,
                args.reasoning_effort,
            )
            row = {
                "db_id": db_id,
                "question": rec["question"],
                "gold_sql": rec["query"],
                "generated_sql": generated_sql,
                "finish_reason": finish_reason,
                "sample_index": args.offset + i,
                "degrade_mode": args.degrade_schema,
                "hidden": hidden,
                "db_path": str(db_path),
            }
            row["is_hallucinated"], row["label_reason"] = label_row(row)
            reasons[row["label_reason"]] += 1
            finishes[finish_reason] += 1
            n_degraded += bool(hidden)

            out_f.write(json.dumps(row) + "\n")
            out_f.flush()
            if i % 25 == 0:
                print(f"[{i}/{len(records)}] {db_id}: {row['label_reason']}")

    print(f"Done. Wrote eval set to {args.out}")
    print("finish_reason counts:", dict(finishes))
    if args.degrade_schema != "none":
        print(f"rows with something hidden from the prompt: {n_degraded}")
    for reason, n in reasons.most_common():
        print(f"  {reason:<28} {n}")


if __name__ == "__main__":
    main()
