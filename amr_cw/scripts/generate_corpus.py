import os
import csv
import json
import time
import random
import hashlib
import argparse

from amr_cw.generation import llm_client
from amr_cw.generation.templates import TEMPLATES, CLASS_TARGET, CLASSES

MODEL_DIR = "models/model_parse_xfm_bart_large-v0_1_0"

SYSTEM_PROMPT = (
    "You write realistic messages for a trust-and-safety research dataset. For each message, invent "
    "a sender persona that naturally fits the situation (a scammer, a debt collector, a coworker, a "
    "friend) and imitate exactly how that sender writes, down to the rhythm and the specifics a real "
    "person would include.\n\n"
    "Rules:\n"
    "- Write a complete, realistic message. No preamble, no subject line, no labels.\n"
    "- This is one continuous message, not a list. Every sentence must add something new: a fresh "
    "detail, a name, a date, an amount, a reference number, a reason, a next step. Never restate a "
    "point you already made in different words.\n"
    "- Ground it in concrete specifics: who is writing, to whom, about what account, order, or "
    "event, with real-sounding names, amounts, dates, and references. The realism comes from the "
    "texture, not from repeating the core ask.\n"
    "- Convey each situation the way a real sender would, woven naturally into the message, never "
    "announcing or naming what you are doing.\n"
    "- Invent fresh, distinct names, organizations, amounts, links, and reference numbers every "
    "time. Do not fall back on a small set of recurring names or numbers across messages.\n"
    "- Write in English only, using plain ASCII punctuation: straight quotes and hyphens, "
    "no em dashes, smart quotes, ellipsis characters, or emoji.\n"
    "- Output ONLY a JSON object, no markdown fences."
)

SEED_CORPUS_PATH = "data/output/real_corpus/_raw/SMSSpamCollection"
CLASS_SEED_LABEL = {'coercive': 'spam', 'deceptive': 'spam', 'benign': 'ham'}

REGISTERS = ['urgent', 'formal', 'casual', 'friendly', 'terse']
PERSONAS = {
    'coercive': ['debt collector', 'angry landlord', 'demanding manager', 'threatening contact'],
    'deceptive': ['fake support agent', 'romance contact', 'phishing sender', 'account impersonator'],
    'benign': ['coworker', 'friend', 'service provider', 'neighbor'],
}

DECOY = {'id': 'DECOY1', 'intent': 'the sender mentions following up later'}
DECOY_FRAC = 0.8

MAX_TOKENS = 800

_PARSE_MODEL = {'stog': None, 'nlp': None}


def load_seed_pools(path):
    pools = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if "\t" not in line:
                continue
            label, text = line.split("\t", 1)
            text = text.strip()
            if text:
                pools.setdefault(label, []).append(text)
    return pools


def build_prompt(class_name, intents, voice, seed_text, decoy):
    lines = [f"Register: {voice['register']} | Sender: {voice['persona']}",
             "",
             "Write one realistic message that naturally works in each of the situations below as "
             "part of a larger, specific message. Weave them in; do not list them and do not name or "
             "label them. Write as a real sender would, with every sentence adding new concrete "
             "detail."]
    if seed_text:
        lines.append(f"\nUse this real message only as a guide to tone, rhythm, and texture; do not "
                     f"copy its content or topic:\n\"{seed_text}\"")
    for i, intent in enumerate(intents):
        lines.append(f"- Situation {i + 1}: {intent}.")
    if decoy:
        lines.append(f"\nAlso include this naturally: {decoy['intent']}.")
    json_fields = ['"text": "..."', '"realizations": [{"item": 1, "span": "..."}]']
    lines.append('\nReturn JSON:\n{' + ', '.join(json_fields) + '}')
    return "\n".join(lines)


def _extract_json_objects(s):
    dec = json.JSONDecoder()
    objs = []
    i, n = 0, len(s)
    while i < n:
        if s[i] == '{':
            try:
                obj, end = dec.raw_decode(s, i)
                objs.append(obj)
                i = end
                continue
            except ValueError:
                pass
        i += 1
    return objs


def parse_response(raw, techniques):
    s = raw.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s
        s = s.rsplit("```", 1)[0]
    s = s.strip()
    try:
        parsed = json.loads(s)
    except json.JSONDecodeError:
        objs = [o for o in _extract_json_objects(s) if isinstance(o, dict) and 'text' in o]
        if not objs:
            raise
        parsed = objs[-1]
    by_pos = {i + 1: t['id'] for i, t in enumerate(techniques)}
    for rz in parsed.get("realizations", []):
        if 'item' in rz and 'technique_id' not in rz:
            rz['technique_id'] = by_pos.get(rz['item'])
    return parsed


def _get_parse_model():
    if _PARSE_MODEL['stog'] is None:
        import amrlib
        from amr_cw.amr_parsing.text_to_graph import load_sentence_splitter
        _PARSE_MODEL['stog'] = amrlib.load_stog_model(MODEL_DIR)
        _PARSE_MODEL['nlp'] = load_sentence_splitter()
    return _PARSE_MODEL['stog'], _PARSE_MODEL['nlp']


def verify_frames(text, techniques):
    from amr_cw.amr_parsing.text_to_graph import text_to_graph
    stog, nlp = _get_parse_model()
    nodes = set(text_to_graph(text, stog, nlp).nodes())
    return [t['id'] for t in techniques if t['amr_frame'] in nodes]


def sample_plan(rng, class_name, techniques_per_class=None):
    class_techniques = TEMPLATES[class_name]
    if techniques_per_class:
        class_techniques = class_techniques[:techniques_per_class]
    count = min(rng.choice([2, 3]), len(class_techniques))
    techniques = rng.sample(class_techniques, count)
    voice = {'register': rng.choice(REGISTERS), 'persona': rng.choice(PERSONAS[class_name])}
    return techniques, voice


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", nargs="+", default=CLASSES)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--corpus", choices=['calibration', 'eval', 'null', 'decoy', 'smoke'], default='calibration')
    ap.add_argument("--techniques-per-class", type=int, default=None)
    ap.add_argument("--test-frac", type=float, default=0.5)
    ap.add_argument("--model", default="claude-opus-4-8")
    ap.add_argument("--seed-corpus", default=SEED_CORPUS_PATH)
    ap.add_argument("--verify-frames", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    backend = os.environ.get('CW_LLM_BACKEND', 'api')
    out_dir = args.out or f"data/output/synthetic_corpus/{args.corpus}"
    sidecar_dir = os.path.join(out_dir, "sidecars")
    raw_dir = os.path.join(out_dir, "raw")
    os.makedirs(sidecar_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{args.corpus}_raw_texts.csv")
    log_path = os.path.join(out_dir, "generation_log.jsonl")

    rng = random.Random(args.seed)
    inject_decoy = (args.corpus == 'decoy')
    seed_pools = load_seed_pools(args.seed_corpus) if args.seed_corpus else {}

    done_ids = set()
    if args.resume:
        done_ids = {f[:-5] for f in os.listdir(sidecar_dir) if f.endswith(".json")}
    if args.resume and args.corpus == 'null':
        raise SystemExit("--resume is unsafe for the null corpus (post-hoc label shuffle needs all docs)")

    sidecars = []
    write_header = not (args.resume and os.path.exists(csv_path))
    csv_mode = "a" if args.resume else "w"
    log_mode = "a" if args.resume else "w"
    with open(csv_path, csv_mode, newline="") as cf, open(log_path, log_mode) as lf:
        writer = csv.writer(cf)
        if write_header:
            writer.writerow(["custom_index", "text", "label", "split", "target"])
            cf.flush()

        for class_name in args.classes:
            for i in range(args.n):
                split = 'test' if rng.random() < args.test_frac else 'train'
                techniques, voice = sample_plan(rng, class_name, args.techniques_per_class)

                use_decoy = inject_decoy and rng.random() < DECOY_FRAC
                decoy = DECOY if use_decoy else None

                seed_pool = seed_pools.get(CLASS_SEED_LABEL.get(class_name, ''), [])
                seed_text = rng.choice(seed_pool) if seed_pool else None

                doc_id = f"{class_name}_{split}_{i}"

                intents = [t['intent'] for t in techniques]
                user_prompt = build_prompt(class_name, intents, voice, seed_text, decoy)

                prompt_hash = hashlib.sha256((SYSTEM_PROMPT + user_prompt).encode()).hexdigest()[:16]

                if doc_id in done_ids:
                    continue

                t0 = time.time()
                raw = None
                usage = {}
                emerged = None
                try:
                    raw, usage = llm_client.complete(
                        SYSTEM_PROMPT, user_prompt, model=args.model, backend=backend,
                        max_tokens=MAX_TOKENS, temperature=None, top_p=None)
                    parsed = parse_response(raw, techniques)
                    text = parsed["text"].strip()
                    realizations = parsed.get("realizations", [])
                    status = "ok"
                except (RuntimeError, KeyError, json.JSONDecodeError, TypeError) as e:
                    status = f"error:{type(e).__name__}"
                    text = None
                    realizations = []

                if status == "ok" and args.verify_frames:
                    try:
                        emerged = verify_frames(text, techniques)
                        if not emerged:
                            status = "discard:no_frame"
                    except Exception as e:
                        status = f"warn:verify_error:{type(e).__name__}"

                raw_record = {"doc_id": doc_id, "status": status, "model": args.model, "backend": backend,
                              "temperature": None, "top_p": None, "max_tokens": MAX_TOKENS,
                              "system_prompt": SYSTEM_PROMPT,
                              "user_prompt": user_prompt, "raw_response": raw, "parsed_text": text}
                with open(os.path.join(raw_dir, f"{doc_id}.json"), "w") as rf:
                    json.dump(raw_record, rf, indent=2)

                log = {"doc_id": doc_id, "status": status, "model": args.model, "backend": backend,
                       "temperature": None, "top_p": None, "max_tokens": MAX_TOKENS,
                       "seed": args.seed, "prompt_hash": prompt_hash,
                       "usage": usage, "secs": round(time.time() - t0, 1)}
                lf.write(json.dumps(log) + "\n")
                lf.flush()

                if status == "discard:no_frame":
                    continue
                if status != "ok":
                    continue

                writer.writerow([doc_id, text, class_name, split, CLASS_TARGET[class_name]])
                cf.flush()

                sidecar = {"doc_id": doc_id, "class": class_name, "split": split,
                           "planted_technique_ids": [t["id"] for t in techniques],
                           "emerged_technique_ids": emerged,
                           "realizations": realizations,
                           "register": voice["register"],
                           "persona": voice["persona"],
                           "seed_text": seed_text,
                           "is_decoy_present": bool(use_decoy),
                           "decoy_id": DECOY["id"] if use_decoy else None}
                sidecar_path = os.path.join(sidecar_dir, f"{doc_id}.json")
                with open(sidecar_path, "w") as sf:
                    json.dump(sidecar, sf, indent=2)
                sidecars.append((sidecar_path, sidecar))

    if args.corpus == 'null':
        shuffle_null_labels(rng, sidecars)


def shuffle_null_labels(rng, sidecars):
    by_class = {}
    for path, sc in sidecars:
        by_class.setdefault(sc["class"], []).append((path, sc))
    for items in by_class.values():
        labels = [sc["planted_technique_ids"] for _, sc in items]
        permuted = labels[:]
        rng.shuffle(permuted)
        for (path, sc), new_label in zip(items, permuted):
            sc["null_realized_technique_ids"] = sc["planted_technique_ids"]
            sc["planted_technique_ids"] = new_label
            sc["null_shuffled"] = True
            with open(path, "w") as sf:
                json.dump(sc, sf, indent=2)


if __name__ == "__main__":
    main()
