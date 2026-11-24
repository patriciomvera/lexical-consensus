r"""Strip LaTeX comments from every .tex file of the arXiv bundle. A whole-line comment is deleted (equivalent for TeX);
an inline comment keeps its % sign (so end-of-line spacing is unchanged) and loses its text. Escaped \% is preserved."""
import re, sys
from pathlib import Path
pat = re.compile(r"(?<!\\)((?:\\\\)*)%.*$")
for p in Path(sys.argv[1]).rglob("*.tex"):
    out = []
    for line in p.read_text().splitlines():
        if re.match(r"^\s*(?<!\\)%", line): continue
        out.append(pat.sub(lambda m: m.group(1) + "%", line))
    p.write_text("\n".join(out) + "\n"); print("stripped", p)
