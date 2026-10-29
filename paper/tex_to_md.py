"""Regenerate docs/ACADEMIC_REPORT.md from paper/forecastkit_phase1.tex so the two never drift apart.

Usage (from the repo root):  python paper/tex_to_md.py
Small, purpose-built converter for this paper only (sections, emphasis, lists, figures, booktabs tables, citations).
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "paper" / "forecastkit_phase1.tex"
BIB = ROOT / "paper" / "references.bib"
OUT = ROOT / "docs" / "ACADEMIC_REPORT.md"
FIG_PREFIX = "figures/eda/"   # relative to docs/


def bib_entries():
    s = BIB.read_text()
    out = {}
    for m in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}", s, re.S):
        f = dict(re.findall(r"(\w+)\s*=\s*\{(.*?)\},?\n", m.group(3) + "\n", re.S))
        out[m.group(2).strip()] = f
    return out


def fmt_ref(f):
    clean = lambda x: re.sub(r"[{}]", "", x or "").replace("--", "-")
    auth = clean(f.get("author")).split(" and ")
    auth = auth[0] if len(auth) == 1 else ", ".join(auth[:-1]) + ", and " + auth[-1]
    parts = [auth + ",", f'"{clean(f.get("title"))},"']
    venue = f.get("journal") or f.get("booktitle") or f.get("publisher")
    if venue:
        parts.append(f"*{clean(venue)}*,")
    if f.get("volume"):
        parts.append(f"vol. {f['volume']},")
    if f.get("number"):
        parts.append(f"no. {f['number']},")
    if f.get("pages"):
        parts.append(f"pp. {clean(f['pages'])},")
    parts.append(f"{f.get('year')}.")
    if f.get("url"):
        parts.append(f.get("url"))
    return " ".join(parts)


def table_to_md(body: str) -> str:
    cap = re.search(r"\\caption\{(.*?)\}\n", body, re.S)
    rows = re.search(r"\\toprule(.*?)\\bottomrule", body, re.S).group(1)
    rows = re.sub(r"\\(midrule|cmidrule\([a-z]*\)\{[^}]*\})", "", rows)
    lines, header_done, md, groups = [l.strip() for l in rows.split("\\\\")], False, [], []
    for l in lines:
        if not l.strip():
            continue
        mc = re.match(r"\\multicolumn\{(\d+)\}\{@\{\}l\}\{(.*)\}$", l)
        if mc:  # full-width group label row
            md.append("| " + " | ".join([f"**{mc.group(2)}**"] + [""] * (int(mc.group(1)) - 1)) + " |")
            continue
        cells = [c.strip() for c in l.split("&")]
        if not header_done and re.search(r"Model|Group|Variant", l) is None:
            groups = [m.group(2) for m in re.finditer(r"\\multicolumn\{(\d+)\}\{[^}]*\}\{(.*?)\}", l)]
            continue  # spanning super-header row: folded into the column names below
        if not header_done and groups:
            n = (len(cells) - 1) // len(groups)
            cells = [cells[0]] + [f"{groups[(i) // n]} {c}" for i, c in enumerate(cells[1:])]
        md.append("| " + " | ".join(cells) + " |")
        if not header_done:
            md.append("|" + "@@SEP@@|" * len(cells))
            header_done = True
    title = inline(cap.group(1)) if cap else ""
    return f"\n*Table: {title}*\n\n" + "\n".join(md) + "\n"


def inline(s: str) -> str:
    s = re.sub(r"\\textbf\{(.*?)\}", r"**\1**", s)
    s = re.sub(r"\\emph\{(.*?)\}", r"*\1*", s)
    s = re.sub(r"\\texttt\{(.*?)\}", lambda m: "`" + m.group(1).replace("\\_", "_") + "`", s)
    s = re.sub(r"\\textsf\{(.*?)\}", r"\1", s)
    s = s.replace("\\%", "%").replace("~", " ").replace("--", "–").replace("``", '"').replace("''", '"')
    s = s.replace("\\,", " ").replace("\\ ", " ").replace("{,}", ",")
    return s


def main():
    tex = TEX.read_text()
    bib = bib_entries()
    body = tex[tex.index("\\begin{document}") + 16: tex.index("\\end{document}")]
    title = re.search(r"\\title\{(.*?)\}\n", body, re.S).group(1).replace("\\\\", " ").replace("\n", " ")
    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", body, re.S).group(1).strip()
    keywords = re.search(r"\\begin\{IEEEkeywords\}(.*?)\\end\{IEEEkeywords\}", body, re.S).group(1).strip()
    main_body = body[body.index("\\end{IEEEkeywords}") + 18:]

    order, labels = [], {}

    def cite(m):
        nums = []
        for k in m.group(1).split(","):
            k = k.strip()
            if k not in order:
                order.append(k)
            nums.append(str(order.index(k) + 1))
        return "[" + ", ".join(nums) + "]"

    # number figures/tables/sections for \ref
    fig_n = tab_n = sec_n = 0
    for m in re.finditer(r"\\(section|begin\{figure\}|begin\{table\*?\}|label)\{?([^}]*)\}?", main_body):
        pass
    for env in re.finditer(r"\\section\{[^}]*\}(\\label\{([^}]*)\})?|\\begin\{(figure|table)\*?\}.*?\\label\{([^}]*)\}|\\subsection\{[^}]*\}\\label\{([^}]*)\}|\\begin\{equation\}.*?\\label\{([^}]*)\}",
                           main_body, re.S):
        if env.group(0).startswith("\\section"):
            sec_n += 1
            if env.group(2):
                labels[env.group(2)] = str(sec_n)
        elif env.group(3) == "figure":
            fig_n += 1
            labels[env.group(4)] = str(fig_n)
        elif env.group(3) == "table":
            tab_n += 1
            labels[env.group(4)] = str(tab_n)
        elif env.group(5):
            labels[env.group(5)] = f"{sec_n}"
        elif env.group(6):
            labels[env.group(6)] = "1"

    s = main_body
    s = re.sub(r"%=+\n", "", s)
    s = re.sub(r"(?m)^%.*\n", "", s)
    s = re.sub(r"\\cite\{([^}]*)\}", cite, s)
    s = re.sub(r"\\eqref\{([^}]*)\}", lambda m: f"({labels.get(m.group(1), '?')})", s)
    s = re.sub(r"\\ref\{([^}]*)\}", lambda m: labels.get(m.group(1), "?"), s)

    def fig(m):
        img = re.search(r"\\includegraphics\[[^]]*\]\{(.*?)\}", m.group(0)).group(1)
        cap = re.search(r"\\caption\{(.*?)\}\n", m.group(0), re.S).group(1)
        lab = re.search(r"\\label\{(.*?)\}", m.group(0)).group(1)
        return f"\n![Fig. {labels[lab]}]({FIG_PREFIX}{img})\n\n*Fig. {labels[lab]}. {inline(cap)}*\n"

    s = re.sub(r"\\begin\{figure\}.*?\\end\{figure\}", fig, s, flags=re.S)
    s = re.sub(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", lambda m: table_to_md(m.group(0)), s, flags=re.S)
    sec_counter = [0]

    def section(m):
        sec_counter[0] += 1
        return f"\n## {sec_counter[0]}. {m.group(1)}\n"

    s = re.sub(r"\\section\{([^}]*)\}(\\label\{[^}]*\})?", section, s)
    s = re.sub(r"\\section\*\{([^}]*)\}", r"\n## \1\n", s)
    s = re.sub(r"\\subsection\{([^}]*)\}(\\label\{[^}]*\})?", r"\n### \1\n", s)
    s = re.sub(r"\\begin\{equation\}(.*?)\\label\{[^}]*\}(.*?)\\end\{equation\}", lambda m: "\n$$" + (m.group(1) + m.group(2)).strip() + "\\tag{1}$$\n", s, flags=re.S)
    s = re.sub(r"\\begin\{(align|gather)\}(.*?)\\end\{\1\}", lambda m: "\n$$\\begin{aligned}" + m.group(2).replace("&", "&").strip() + "\\end{aligned}$$\n", s, flags=re.S)
    s = re.sub(r"\\begin\{(itemize|enumerate)\}|\\end\{(itemize|enumerate)\}", "", s)
    s = re.sub(r"\\item\s*", "- ", s)
    s = s.replace("\\appendix", "")
    s = re.sub(r"\\bibliographystyle\{[^}]*\}\s*\\bibliography\{[^}]*\}", "%%REFS%%", s)
    s = inline(s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    refs = "\n## References\n\n" + "\n".join(f"{i + 1}. {fmt_ref(bib[k])}" for i, k in enumerate(order)) + "\n"
    s = s.replace("%%REFS%%", refs).replace("@@SEP@@", "---")

    head = (f"<!-- Generated from paper/forecastkit_phase1.tex by paper/tex_to_md.py. Edit the .tex, then regenerate. -->\n\n"
            f"# {inline(title)}\n\n**Anonymous submission** · Phase 1 (Machine Learning) project report\n\n"
            f"PDF: `paper/forecastkit_phase1.pdf` · Evidence: `docs/PHASE1_EVIDENCE_INDEX.md`\n\n"
            f"### Abstract\n\n{inline(abstract)}\n\n**Keywords:** {inline(keywords)}\n")
    OUT.write_text(head + s.strip() + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}  ({len(order)} references, {fig_n} figures, {tab_n} tables)")


if __name__ == "__main__":
    main()
