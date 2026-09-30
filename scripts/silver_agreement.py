"""Agreement between two independent silver annotations of the same papers.

    python scripts/silver_agreement.py data/annotations_silver data/annotations_silver_b

Both annotations were made by LLM sub-agents with the same guideline (docs/SILVER_GUIDELINE.md) without
seeing each other. High agreement shows the silver standard is *consistent*; it does not prove it is
*correct* (both annotators share the same model biases). A human-verified subset is still needed for that.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import PRF, _k, relaxed_relation_prf  # noqa: E402


def main(a_dir: str, b_dir: str) -> None:
    tot = {k: PRF() for k in ("entities", "relations_strict", "relations_relaxed", "values")}
    stems = sorted(p.stem for p in Path(b_dir).glob("*.json") if (Path(a_dir) / p.name).exists())
    for stem in stems:
        a = json.loads((Path(a_dir) / f"{stem}.json").read_text(encoding="utf-8"))
        b = json.loads((Path(b_dir) / f"{stem}.json").read_text(encoding="utf-8"))
        tot["entities"].add({(e["type"], _k(e["canonical"])) for e in a["entities"]},
                            {(e["type"], _k(e["canonical"])) for e in b["entities"]})
        tot["relations_strict"].add({(_k(r["subject"]), r["relation"], _k(r["object"])) for r in a["relations"]},
                                    {(_k(r["subject"]), r["relation"], _k(r["object"])) for r in b["relations"]})
        rx = relaxed_relation_prf([(r["subject"], r["relation"], r["object"]) for r in a["relations"]],
                                  [(r["subject"], r["relation"], r["object"]) for r in b["relations"]])
        t = tot["relations_relaxed"]
        t.tp, t.fp, t.fn = t.tp + rx.tp, t.fp + rx.fp, t.fn + rx.fn

        def vk(v):
            return (_k(v["entity"]), v.get("normalized_value"), v.get("normalized_min"), v.get("normalized_max"),
                    v["normalized_unit"])
        tot["values"].add({vk(v) for v in a["values"]}, {vk(v) for v in b["values"]})
    print(f"Agreement on {len(stems)} papers ({', '.join(stems)}); A = reference")
    print("| Task | P | R | F1 |\n|---|---|---|---|")
    for k, v in tot.items():
        s = v.summary()
        print(f"| {k} | {s['precision']} | {s['recall']} | {s['f1']} |")


if __name__ == "__main__":
    main(*sys.argv[1:3])
