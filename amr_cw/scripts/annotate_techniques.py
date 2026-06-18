import os
import csv
import json
import time
import math
import argparse

from amr_cw.generation import llm_client
from amr_cw.generation.templates import TEMPLATES

csv.field_size_limit(10 ** 7)

SYSTEM_PROMPT = (
    "You are a trust-and-safety reviewer labeling which manipulation techniques appear in a short "
    "message. You are given a menu of candidate techniques for the message's category. Decide which "
    "techniques the TEXT actually shows - judge only what is written, not what might be implied.\n"
    "Return ONLY JSON: {\"technique_ids\": [\"...\"]}. Use [] if none are clearly present."
)


def normalize_technique_ids(annotated, valid):
    valid_set = set(valid)
    out = []
    for a in annotated:
        if not isinstance(a, str) or not a.strip():
            continue
        token = a.strip().split()[0].rstrip(":.,")
        if token in valid_set and token not in out:
            out.append(token)
    return out


def technique_menu(class_name):
    lines = [f"Category: {class_name}", "Candidate techniques:"]
    for t in TEMPLATES[class_name]:
        lines.append(f"- {t['id']} {t['name']}: {t['intent']}.")
    return "\n".join(lines)


def build_user_prompt(class_name, text):
    return f"{technique_menu(class_name)}\n\nMessage:\n\"{text}\"\n\nReturn JSON {{\"technique_ids\": [...]}}."


def load_corpus(csv_path):
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def load_planted(sidecar_dir, doc_id):
    path = os.path.join(sidecar_dir, f"{doc_id}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)["planted_technique_ids"]


def wilson_interval(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def drift_report(records):
    by_class = {}
    for rec in records:
        by_class.setdefault(rec["class"], []).append(rec)

    report = {}
    for class_name, recs in by_class.items():
        labeled = [r for r in recs if r["planted_technique_ids"] is not None]
        n = len(labeled)
        if n == 0:
            continue
        exact = sum(1 for r in labeled if set(r["annotated_technique_ids"]) == set(r["planted_technique_ids"]))
        jaccards = []
        confirmed = total_planted = 0
        for r in labeled:
            planted = set(r["planted_technique_ids"])
            annotated = set(r["annotated_technique_ids"])
            union = planted | annotated
            jaccards.append(len(planted & annotated) / len(union) if union else 1.0)
            confirmed += len(planted & annotated)
            total_planted += len(planted)
        drift = 1 - exact / n
        lo, hi = wilson_interval(n - exact, n)
        report[class_name] = {
            "n": n,
            "exact_match_rate": round(exact / n, 3),
            "drift_rate": round(drift, 3),
            "drift_ci95": [round(lo, 3), round(hi, 3)],
            "mean_jaccard": round(sum(jaccards) / n, 3),
            "planted_technique_confirm_rate": round(confirmed / total_planted, 3) if total_planted else None,
        }
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="eval")
    ap.add_argument("--in", dest="csv_path", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--model", default="gpt-4.1-2025-04-14")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--import-labels", dest="import_labels", default=None)
    args = ap.parse_args()

    base = f"data/output/synthetic_corpus/{args.corpus}"
    csv_path = args.csv_path or os.path.join(base, f"{args.corpus}_raw_texts.csv")
    sidecar_dir = os.path.join(base, "sidecars")
    raw_dir = os.path.join(base, "annotation_raw")
    out_path = args.out or os.path.join(base, "annotations.json")
    drift_path = os.path.join(base, "drift_report.json")
    log_path = os.path.join(base, "annotation_log.jsonl")

    rows = load_corpus(csv_path)

    if args.export:
        export_path = os.path.join(base, "to_annotate.jsonl")
        with open(export_path, "w") as ef:
            for row in rows:
                ef.write(json.dumps({"doc_id": row["custom_index"], "class": row["label"],
                                     "text": row["text"], "menu": technique_menu(row["label"])}) + "\n")
        return

    imported = None
    if args.import_labels:
        imported = {}
        with open(args.import_labels) as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    imported[obj["doc_id"]] = obj["technique_ids"]

    os.makedirs(raw_dir, exist_ok=True)
    records = []
    with open(log_path, "w") as lf:
        for i, row in enumerate(rows):
            doc_id, class_name, text = row["custom_index"], row["label"], row["text"]
            user_prompt = build_user_prompt(class_name, text)
            t0 = time.time()
            raw = None
            if imported is not None:
                annotated = imported.get(doc_id, [])
                usage, status = {}, "imported"
            else:
                try:
                    raw, usage = llm_client.openai_complete(
                        SYSTEM_PROMPT, user_prompt, model=args.model, temperature=args.temperature)
                    annotated = json.loads(raw).get("technique_ids", [])
                    status = "ok"
                except (RuntimeError, json.JSONDecodeError, KeyError) as e:
                    annotated, usage, status = [], {}, f"error:{type(e).__name__}"

            raw_record = {"doc_id": doc_id, "class": class_name, "status": status, "model": args.model,
                          "system_prompt": SYSTEM_PROMPT, "user_prompt": user_prompt, "raw_response": raw}
            with open(os.path.join(raw_dir, f"{doc_id}.json"), "w") as rf:
                json.dump(raw_record, rf, indent=2)

            valid = [t["id"] for t in TEMPLATES[class_name]]
            annotated = normalize_technique_ids(annotated, valid)
            planted = load_planted(sidecar_dir, doc_id)
            records.append({"doc_id": doc_id, "class": class_name,
                            "annotated_technique_ids": annotated, "planted_technique_ids": planted})
            lf.write(json.dumps({"doc_id": doc_id, "status": status, "model": args.model,
                                 "usage": usage, "secs": round(time.time() - t0, 1)}) + "\n")
            lf.flush()

    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)

    report = drift_report(records)
    with open(drift_path, "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
