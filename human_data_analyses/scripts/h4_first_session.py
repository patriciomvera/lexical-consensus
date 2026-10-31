"""H4 (DESV §29): first real participation per person, from XTMEM all_raw_A.csv (commit 220ed17).
Earliest AcceptTime per subid (SubmitTime gives the same choice). Writes data/h4_first_session.csv (subid, first_exp).
This file is a per-participant derivative of third-party data and is NOT redistributed; regenerate it locally."""
import sys, pathlib as _pl; sys.path.insert(0, str(_pl.Path(__file__).resolve().parent)); import _paths as _P
import pathlib, pandas as pd
B = pathlib.Path(_P.HDA)
r = pd.read_csv(B / "data/xtmem/all_raw_A.csv", usecols=["exp", "subids", "AcceptTime"])
r["t"] = pd.to_datetime(r.AcceptTime, utc=True)
f = r.sort_values(["subids", "t", "exp"]).groupby("subids", as_index=False).first()
f = f.rename(columns={"subids": "subid", "exp": "first_exp"})[["subid", "first_exp"]].astype(int).sort_values("subid")
f.to_csv(B / "data/h4_first_session.csv", index=False)
print(f.shape)
