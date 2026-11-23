"""Build the manuscript exhibits from the archived item decisions and model fits."""
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
V2 = HERE.parent
FROZEN = V2 / "encoder_equivalence_v2" / "encoder_equivalence"
MATCHED = V2 / "agents_xtmem_v2" / "outputs"
NAMES = {
    "adult": "Adults",
    "R0_dinov2s": "DINOv2-small",
    "O1_dinov2b": "DINOv2-base",
    "R1_cvcl": "CVCL",
    "E1_siglip": "SigLIP",
    "O2_openclip": "OpenCLIP",
    "E2_vitsup": "Supervised ViT",
    "E3_rn50sup": "Supervised ResNet-50",
    "E4_mae": "MAE",
    "F1_rn50rand0": "Random ResNet-50 (s0)",
    "F2_pixels": "Pixels",
    "E4s_mae_cls": "MAE (CLS variant)",
    "F1_rn50rand1": "Random ResNet-50 (s1)",
    "F1_rn50rand2": "Random ResNet-50 (s2)",
}
PRIMARY = list(NAMES)[1:11]
CONDITIONS = ["3sub", "3bas", "3sup"]
DOMAINS = {"animals": "c3", "vegetables": "c1", "vehicles": "c2"}
SETS = ["within", "outside"]
TEAL, ORANGE, INK = "#17615c", "#a64c24", "#173446"
mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42,
                     "text.color": INK, "axes.labelcolor": INK, "ytick.color": INK})


def matched_cells():
    items = pd.read_csv(MATCHED / "one_word_items.csv")
    items = items[(items.learner == "centroid") & items.condition.isin(CONDITIONS)].copy()
    key = ["domain", "condition", "image"]
    assert set(items.encoder) == set(NAMES) - {"adult"}
    assert not items.duplicated(["encoder"] + key).any()
    assert (items.groupby(key)[["human_yes", "human_n"]].nunique() == 1).all().all()
    assert items.human_n.eq(600).all() and items.accepted.isin([0, 1]).all()
    own = items.image.str[:2].eq(items.domain.map(DOMAINS))
    items["candidate_set"] = np.where(own, "within", "outside")
    items = items[~own | items.image.str.contains("_sup")].copy()
    people = items[items.encoder == PRIMARY[0]].assign(encoder="adult")
    people["yes"], people["total"] = people.human_yes, people.human_n
    agents = items.assign(yes=items.accepted.astype(int), total=1)
    return pd.concat([people, agents], ignore_index=True)


def aggregate(items, dimension, levels, expected_counts):
    keys = ["encoder", "candidate_set", dimension]
    cells = items.groupby(keys).agg(yes=("yes", "sum"), total=("total", "sum"),
                                   candidate_count=("image", "size"))
    rows = pd.MultiIndex.from_product([NAMES, SETS, levels], names=keys)
    cells = cells.reindex(rows)
    assert not cells.isna().any().any()
    for subset, count in zip(SETS, expected_counts):
        assert cells.xs(subset, level="candidate_set").candidate_count.eq(count).all()
    cells["percent"] = 100 * cells.yes / cells.total
    return cells


def distance_effects():
    controls = pd.read_csv(FROZEN / "results/poscontrol.csv").query("term == 'zd'").set_index("encoder")
    rows = []
    for encoder in PRIMARY:
        human = controls.loc[encoder]
        path = FROZEN / "agents/out/results" / f"D_{encoder}.json"
        agent = json.loads(path.read_text())["learners"]["centroid"]["model_ps"]
        assert human["items"] == 8 and agent["n"] == 100 and agent["status"] == "ok"
        rows.append([encoder, human.estimate, human.zd_lo, human.zd_hi,
                     agent["r_partial"], agent["r_ci_lo"], agent["r_ci_hi"]])
    return pd.DataFrame(rows, columns=["encoder", "adult_beta", "adult_low", "adult_high",
                                       "agent_r", "agent_low", "agent_high"]).set_index("encoder")


def percentage(value, row):
    if value in (0, 100):
        return str(int(value))
    return f"{value:.2f}" if row == "adult" and value < 10 else f"{value:.1f}"


def wide_table(cells, levels):
    return cells.percent.unstack(["candidate_set", cells.index.names[-1]]).loc[
        list(NAMES), pd.MultiIndex.from_product([SETS, levels])]


def write_latex(table):
    lines = [
        r"% Requires booktabs. Include this file from a one- or two-column manuscript.",
        r"\begin{table*}[t]", r"\centering", r"\small",
        r"\caption{Extension of a novel word after three teaching examples. Entries are percentages.}",
        r"\label{tab:matched-extension}", r"\begin{tabular}{lrrrrrr}", r"\toprule",
        r" & \multicolumn{3}{c}{Within-domain superordinates} & \multicolumn{3}{c}{Outside domain} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r" & Subtype & Basic & Broad & Subtype & Basic & Broad \\", r"\midrule",
    ]
    for enc, values in table.iterrows():
        if enc in [PRIMARY[0], PRIMARY[8], "E4s_mae_cls"]:
            lines.append(r"\midrule")
        lines.append(" & ".join([NAMES[enc]] + [percentage(v, enc) for v in values]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\par\smallskip",
              r"\begin{minipage}{0.96\linewidth}\footnotesize",
              "Subtype, Basic and Broad denote three examples of the same subtype, of different subtypes of one basic type, and of different basic types in one domain, respectively. "
              "Each condition has 12 within-domain superordinate and 48 outside-domain word--image combinations, weighted equally across domains. "
              "The adult row pools 600 judgments per combination from a study with 520 unique adults. Agent rows give the fraction accepted by the centroid learner with a support-dependent rule. "
              "Blocks below the adult row show eight trained encoders, two controls and three exploratory variants. s0--s2 denote network initialization seeds.",
              r"\end{minipage}", r"\end{table*}"]
    (HERE / "word_extension_table.tex").write_text("\n".join(lines) + "\n")


def heatmap(table, headings, title, subtitle, notes):
    fig = plt.figure(figsize=(11.6, 8.5))
    fig.text(.04, .94, title, fontsize=20, weight="bold")
    fig.text(.04, .895, subtitle, fontsize=11.5)
    y = [0, *np.arange(1, 9)+.55, *np.arange(9, 11)+1.10, *np.arange(11, 14)+1.65]
    for j, (x, label, colors) in enumerate([
        (.245, "Within-domain superordinates", ["#f1f7f6", "#8dc7c0", "#095955"]),
        (.625, "Images from other domains", ["#fcf7f1", "#e9bd99", "#944021"]),
    ]):
        ax = fig.add_axes([x, .155, .335, .625])
        ax.set(xlim=(-.5, 2.5), ylim=(y[-1]+.55, -.55))
        ax.set_axis_off()
        fig.text(x, .836, label, color=colors[-1], size=13, weight="bold")
        cmap = LinearSegmentedColormap.from_list("rates", colors)
        for k, heading in enumerate(headings):
            ax.text(k, -.72, heading, ha="center", va="bottom", size=10, clip_on=False)
        for (enc, row), position in zip(table.iterrows(), y):
            if j == 0:
                ax.text(-.57, position, NAMES[enc], ha="right", va="center",
                        weight="bold" if enc == "adult" else "normal")
            for k, v in enumerate(row.iloc[3*j:3*j+3]):
                color = cmap(v / 100)
                ax.add_patch(plt.Rectangle((k-.5, position-.47), 1, .94,
                                           color=color, ec="white", lw=.7))
                rgb = np.array(color[:3])
                luminance = np.where(rgb <= .04045, rgb/12.92, ((rgb+.055)/1.055)**2.4) @ [.2126,.7152,.0722]
                ax.text(k, position, percentage(v, enc), ha="center", va="center",
                        color="white" if luminance < .20 else INK,
                        weight="bold" if enc == "adult" else "normal")
        if j == 0:
            for pos, label in [(.79,"TRAINED"), (9.32,"CONTROLS"), (11.86,"VARIANTS")]:
                ax.text(-.57, pos, label, ha="right", va="center", fontsize=7, color="#526978", weight="bold")
    for i, note in enumerate(notes):
        fig.text(.04, .106-.030*i, note, size=9.3)
    return fig


def forest(effects):
    fig = plt.figure(figsize=(11.6, 7.8))
    fig.text(.04, .935, "How well does visual distance predict behavior?", fontsize=20, weight="bold")
    fig.text(.04, .885, "The adult task measures word extension. The agent task measures naming accuracy on CIFAR-100.", size=12)
    y = np.r_[np.arange(8), np.arange(8, 10)+.45]
    for x, title, metric, low, high, limits, color, xlabel in [
        (.24, "Adult word extension", "adult_beta", "adult_low", "adult_high", (-4.8,.5), TEAL, "Log-odds change per SD of distance"),
        (.635, "Agent naming accuracy", "agent_r", "agent_low", "agent_high", (-.82,.41), ORANGE, "Partial correlation with visual distance"),
    ]:
        fig.text(x,.80,title,weight="bold",size=14,color=color)
        ax = fig.add_axes([x,.24,.32,.51])
        ax.set(xlim=limits, ylim=(y[-1]+.6,-.55), xlabel=xlabel)
        ax.set_axisbelow(True)
        ax.grid(axis="x",color="#dbe3e5")
        ax.axvline(0, color="#657785", ls="--", lw=1)
        ax.axhline(7.7, color="#bbc9ce", lw=1)
        ax.errorbar(effects[metric], y, xerr=[effects[metric]-effects[low], effects[high]-effects[metric]],
                    fmt="o", color=color, markersize=6, capsize=3, lw=1.6)
        ax.set_yticks(y, [NAMES[e] for e in effects.index] if metric == "adult_beta" else [])
        ax.tick_params(length=0, pad=6)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.text(.04,.128,"Adult fits use the same responses to eight within-domain items and adjust for taxonomic level and teaching condition.",size=10)
    fig.text(.04,.092,"Agent fits use the 100 original class pairs and adjust for WordNet distance. Bars are the archived 95% intervals.",size=10)
    fig.text(.04,.056,"The two panels have different units. Negative values indicate lower selection or accuracy at greater distance.",size=10)
    return fig


def main():
    items = matched_cells()
    conditions = aggregate(items, "condition", CONDITIONS, [12, 48])
    domains = aggregate(items[items.condition == "3sup"], "domain", list(DOMAINS), [4, 16])
    effects = distance_effects()
    for name, frame in [("teaching_conditions", conditions), ("broad_teaching_by_domain", domains),
                        ("distance_effects", effects)]:
        frame.to_csv(HERE / f"{name}.csv")
    overall = wide_table(conditions, CONDITIONS)
    write_latex(overall)
    figures = [
        heatmap(overall, ["Same subtype","Same basic type","Broad teaching"],
                "When do adults and agents extend a word?", "Percentage selected or accepted after three teaching examples. Each condition uses the same test images.",
                ["Within each teaching condition, there are 12 within-domain superordinates and 48 outside-domain word–image combinations.",
                 "The adult row gives selection frequencies. Agent rows give fractions of candidates accepted by a centroid learner using support dispersion.",
                 "Domains are equally weighted. Adults contributed 600 responses per combination across 520 unique participants."]),
        heatmap(wide_table(domains, list(DOMAINS)), ["Animals","Vegetables","Vehicles"],
                "Where does extension stop?", "Percentage selected or accepted after broad teaching. Columns identify the taught domain.",
                ["Broad teaching uses three examples drawn from different basic types in the same domain.",
                 "Each domain contributes four within-domain superordinates and 16 outside-domain candidates. One agent decision changes the within-domain rate by 25 points.",
                 "Adult percentages pool 600 responses per candidate. s0, s1 and s2 are initialization seeds for the random network."]),
        forest(effects),
    ]
    with PdfPages(HERE / "adults_and_agents_by_encoder.pdf",
                  metadata={"Title":"Adults and agents across visual encoders","CreationDate":None,"ModDate":None}) as pdf:
        for fig in figures:
            pdf.savefig(fig)
            plt.close(fig)


if __name__ == "__main__":
    main()
