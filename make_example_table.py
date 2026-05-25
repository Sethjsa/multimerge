"""Generate a LaTeX booktabs table for one example from qualitative_eng_fra.jsonl."""

import json
import sys

JSONL = "qualitative_eng_fra.jsonl"
MAX_CHARS = 200

MODEL_LABELS = {
    "tiny-aya":   r"\textsc{Tiny-Aya}",
    "hplt2c-eng": r"\textsc{HPLT-Eng}",
    "hplt2c-fra": r"\textsc{HPLT-Fra}",
    "ties":       r"\textsc{Ties}",
    "dareties":   r"\textsc{DareTies}",
    "mixed":      r"\textsc{Mixed}",
    "merged":     r"\textsc{Merged}",
}

SPECIALS = [
    ("\\", r"\textbackslash{}"),
    ("&",  r"\&"),
    ("%",  r"\%"),
    ("$",  r"\$"),
    ("#",  r"\#"),
    ("_",  r"\_"),
    ("{",  r"\{"),
    ("}",  r"\}"),
    ("~",  r"\textasciitilde{}"),
    ("^",  r"\textasciicircum{}"),
]

def escape(s: str) -> str:
    for ch, rep in SPECIALS:
        s = s.replace(ch, rep)
    return s

def clean(s: str) -> str:
    first_line = s.split("\n")[0].strip()
    if len(first_line) > MAX_CHARS:
        return escape(first_line[:MAX_CHARS]) + r"\ldots{}"
    return escape(first_line)

def main():
    lineno = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    with open(JSONL) as f:
        for i, line in enumerate(f, 1):
            if i == lineno:
                r = json.loads(line)
                break
        else:
            sys.exit(f"Line {lineno} not found in {JSONL}")

    rows = [
        (r"\textit{Source}",    clean(r["src"])),
        (r"\textit{Reference}", clean(r["tgt"])),
    ]
    for key, label in MODEL_LABELS.items():
        if key in r:
            rows.append((label, clean(r[key])))

    print(r"\begin{table}[t]")
    print(r"  \centering")
    print(r"  \small")
    print(r"  \begin{tabular}{lp{9cm}}")
    print(r"    \toprule")
    print(r"    \textbf{System} & \textbf{Translation} \\")
    print(r"    \midrule")
    for i, (label, text) in enumerate(rows):
        if i == 2:
            print(r"    \midrule")
        print(f"    {label} & {text} \\\\")
    print(r"    \bottomrule")
    print(r"  \end{tabular}")
    print(r"  \caption{Example translations (FLORES devtest, English$\to$French, example " + str(lineno) + r").}")
    print(r"\end{table}")

if __name__ == "__main__":
    main()
