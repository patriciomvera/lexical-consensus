"""Observed (not bootstrap-mean) Q4-Q3 median difference, centroid learner, original quadrants."""
import sys, pandas as pd
d = pd.read_csv(f"{sys.argv[1] if len(sys.argv) > 1 else 'results/exp_008_dissociation'}/phase3/exp_008_c1_results.csv"); d = d[d.learner == "centroid"]
m = d.groupby("quadrant").c1_accuracy.median(); print(m.to_dict()); print("observed Q4-Q3 median diff", round(m["Q4"] - m["Q3"], 4))
