"""v2c table edits applied after generation (asserted, idempotent-safe on fresh copies). Usage: python fix_tables_v2c.py <manuscrito_v2c/tables>"""
import sys
from pathlib import Path
T = Path(sys.argv[1])
def R(fn, old, new):
    p = T / fn; s = p.read_text(); assert s.count(old) == 1, (fn, old); p.write_text(s.replace(old, new))
R("tab2a_agents_distance.tex", r"& Region D & D-all: $r$ & Region D-all", r"& Region D & D-all: partial $r$ & Region D-all")
# Mantel p to three decimals, except .0505, which would round onto the .05 boundary
R("tab2b_agents_human.tex", r"& .0148 & I/N/N", r"& .015 & I/N/N")
R("tab2b_agents_human.tex", r"& .0056 & I/I/I", r"& .006 & I/I/I")
print("tables fixed")
