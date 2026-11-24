"""v2f fixes to the generated post hoc appendix and Table 13 (A. review). Usage: python fix_appendix_tables_v2f.py <manuscrito_v2f>"""
import sys
from pathlib import Path
H = Path(sys.argv[1])
def R(p, old, new):
    s = p.read_text(); assert s.count(old) == 1, (p, old[:80]); p.write_text(s.replace(old, new))
R(H / "tables/tab_posthoc_v2c.tex", r"& Frozen model & By-image slopes &", r"& Frozen model (refit) & By-image slopes &")
a = H / "appendix_rules_posthoc_v2c.tex"
R(a, "after two external reviews of an earlier draft", "after two automated external reviews of an earlier draft")
R(a, r"(frozen) or (1 + $z_d$~$\mid$~person) + (1 + $z_d$~$\mid$~image) (by-image slopes).",
     r"(frozen) or (1 + $z_d$~$\mid$~person) + (1 + $z_d$~$\mid$~image) (by-image slopes). Intervals are from the post hoc refit and may differ from Table~\ref{tab:v2adults} in the second decimal.")
print("appendix and Table 13 fixed")
