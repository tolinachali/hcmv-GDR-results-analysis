# hcmv_gdr_circos_bars_balanced_highlight_export.py
import os
import pandas as pd
from Bio import SeqIO
from pycirclize import Circos
from matplotlib.patches import Patch

# -----------------------------
# User inputs
# -----------------------------
gb_file = "/home/tolina/Desktop/hcmv_manuscript_data/project_folder/results/NC_006273.gb"
gdr_file = "/home/tolina/Desktop/hcmv_manuscript_data/project_folder/window/strain_type/GDRcalculationResultsJ/hcmv_gdr__dist_mat_list__Strain_Type__pvals.tsv"
fasta_dir = "/home/tolina/Desktop/hcmv_manuscript_data/project_folder/window/parsed_xml_out"
strain_map_file = "/home/tolina/Desktop/CMV_data/analysis_v1/all_willowj/pass_willowj/strain_.tsv"
output_png = "hcmv_gdr_circos_bars_balanced_highlight.png"
output_table = "gdr_window_qc_balanced.tsv"

# Parameters
pval_cutoff = 0.05
gdr_cutoff = 0.8
smaller_group_fraction = 0.5  # 50% of larger group

# radial track definitions
TRACK_OVERALL = (50, 60)
TRACK_CLINICAL = (65, 75)
TRACK_LAB = (80, 90)
TRACK_HIGHLIGHT = (40, 48)
TRACK_GENES = (92, 96)
TRACK_AXIS = (98, 100)

# Label stacking
BASE_LABEL_RADIUS = TRACK_HIGHLIGHT[0] - 2
LABEL_STEP = 2

# -----------------------------
# Load genome and GDR
# -----------------------------
record = SeqIO.read(gb_file, "genbank")
genome_len = len(record.seq)

gdr_df = pd.read_csv(gdr_file, sep="\t", dtype=object)
strain_map = pd.read_csv(strain_map_file, sep="\t")
sample_to_type = dict(zip(strain_map["Sample_ID"], strain_map["Strain_Type"]))

# Prepare Start/End columns
if "Start" not in gdr_df.columns or "End" not in gdr_df.columns:
    n_windows = len(gdr_df)
    window_size = int(round(genome_len / n_windows))
    gdr_df["Start"] = [i * window_size for i in range(n_windows)]
    gdr_df["End"] = [min(genome_len, (i * window_size) + window_size) for i in range(n_windows)]

gdr_df["Start"] = pd.to_numeric(gdr_df["Start"], errors="coerce").astype("Int64")
gdr_df["End"] = pd.to_numeric(gdr_df["End"], errors="coerce").astype("Int64")

# Convert numeric columns
for col in gdr_df.columns:
    if "CDR" in col or "p-value" in col:
        gdr_df[col] = pd.to_numeric(gdr_df[col], errors="coerce")

gdr_df = gdr_df.dropna(subset=["Start", "End"]).reset_index(drop=True)


# -----------------------------
# Compute n_c, n_l and balance
# -----------------------------
def compute_sample_counts(window_feature):
    fasta_base = window_feature.replace("_aic_distmax", "")
    fasta_file = os.path.join(fasta_dir, f"{fasta_base}.fasta")
    try:
        samples = [rec.id for rec in SeqIO.parse(fasta_file, "fasta")]
        n_c = sum(1 for s in samples if sample_to_type.get(s) == "clinical")
        n_l = sum(1 for s in samples if sample_to_type.get(s) == "lab")
    except FileNotFoundError:
        n_c, n_l = 0, 0
    return n_c, n_l


n_c_list, n_l_list, balanced_list, balanced_flag_list = [], [], [], []
for feature in gdr_df["Feature"]:
    n_c, n_l = compute_sample_counts(feature)
    n_c_list.append(n_c)
    n_l_list.append(n_l)
    if max(n_c, n_l) == 0:
        balanced = False
    else:
        balanced = min(n_c, n_l) >= smaller_group_fraction * max(n_c, n_l)
    balanced_list.append(balanced)
    balanced_flag_list.append("balanced" if balanced else "imbalanced")

gdr_df["n_c"] = n_c_list
gdr_df["n_l"] = n_l_list
gdr_df["balanced"] = balanced_list
gdr_df["balance_flag"] = balanced_flag_list  # for export

# -----------------------------
# Filter significant windows
# -----------------------------
df_sig = gdr_df[
    gdr_df["Strain_Type(p-value)"] < pval_cutoff].copy() if "Strain_Type(p-value)" in gdr_df.columns else gdr_df.copy()

# -----------------------------
# GDR columns
# -----------------------------
col_overall = "Strain_Type(CDR)"
col_clin = "clinical(CDR)"
col_lab = "lab(CDR)"
for c in [col_overall, col_clin, col_lab]:
    if c not in gdr_df.columns:
        gdr_df[c] = 0.0
    gdr_df[c] = pd.to_numeric(gdr_df[c], errors="coerce").fillna(0.0)

max_overall = gdr_df[col_overall].max() or 1.0
max_clin = gdr_df[col_clin].max() or 1.0
max_lab = gdr_df[col_lab].max() or 1.0

# -----------------------------
# Build Circos
# -----------------------------
circos = Circos(sectors={"HCMV": genome_len})
sector = circos.sectors[0]

# Axis
axis_track = sector.add_track(TRACK_AXIS)
axis_track.axis()
axis_track.xticks_by_interval(20000, label_formatter=lambda v: f"{v / 1000:.0f} kb")
axis_track.xticks_by_interval(5000, tick_length=0.5, show_label=False)

# Gene track
gene_track = sector.add_track(TRACK_GENES)
for feat in record.features:
    if feat.type == "gene":
        s, e = int(feat.location.start), int(feat.location.end)
        gene_track.rect(s, e, fc="lightgrey", ec="none")
        if (e - s) > 800:
            name = feat.qualifiers.get("gene", [""])[0] if "gene" in feat.qualifiers else \
            feat.qualifiers.get("locus_tag", [""])[0]
            if name:
                gene_track.text(text=name, x=(s + e) // 2, r=None, size=5, orientation="vertical", color="black")


# Helper for radial bar
def draw_bar(track, start, end, value, max_value, r_min, r_max, color):
    frac = (value / max_value) if max_value != 0 else 0.0
    r_end = r_min + max(0.0, min(1.0, frac)) * (r_max - r_min)
    track.rect(start, end, r_lim=(r_min, r_end), fc=color, ec="none")


# Tracks
clin_track = sector.add_track(TRACK_CLINICAL)
lab_track = sector.add_track(TRACK_LAB)
overall_track = sector.add_track(TRACK_OVERALL)

# Draw bars
df_plot = df_sig if len(df_sig) > 0 else gdr_df
for _, row in df_plot.iterrows():
    s, e = int(row["Start"]), int(row["End"])
    draw_bar(clin_track, s, e, row[col_clin], max_clin, TRACK_CLINICAL[0], TRACK_CLINICAL[1], "steelblue")
    draw_bar(lab_track, s, e, row[col_lab], max_lab, TRACK_LAB[0], TRACK_LAB[1], "tomato")
    draw_bar(overall_track, s, e, row[col_overall], max_overall, TRACK_OVERALL[0], TRACK_OVERALL[1], "grey")

# -----------------------------
# Highlight low-GDR windows and mark imbalance
# -----------------------------
highlight_track = sector.add_track(TRACK_HIGHLIGHT)
label_positions = []


def get_staggered_radius(x_start, x_end):
    radius = BASE_LABEL_RADIUS
    collision = True
    while collision:
        collision = False
        for (sx, ex, r) in label_positions:
            if not (x_end < sx or x_start > ex):
                if abs(radius - r) < LABEL_STEP:
                    collision = True
                    radius -= LABEL_STEP
                    break
    label_positions.append((x_start, x_end, radius))
    return radius


for _, row in df_plot.iterrows():
    s, e = int(row["Start"]), int(row["End"])
    if row[col_overall] <= gdr_cutoff:
        color = "darkred" if row["balanced"] else "orange"
        highlight_track.rect(s, e, fc=color, ec="none")

        if row["balanced"]:
            overlapping = [
                feat.qualifiers.get("gene", [""])[0] if "gene" in feat.qualifiers else
                feat.qualifiers.get("locus_tag", [""])[0]
                for feat in record.features if
                feat.type == "gene" and (s <= int(feat.location.end) and e >= int(feat.location.start))
            ]
            if overlapping:
                label = overlapping[0]
                r_label = get_staggered_radius(s, e)
                highlight_track.text(text=label, x=(s + e) // 2, r=r_label, size=6, orientation="vertical",
                                     color="darkgreen")

# -----------------------------
# Legend
# -----------------------------
handles = [
    Patch(color="lightgrey", label="Annotated genes"),
    Patch(color="steelblue", label="Clinical GDR (bar length)"),
    Patch(color="tomato", label="Lab GDR (bar length)"),
    Patch(color="grey", label="Overall GDR (bar length)"),
    Patch(color="darkred", label="Low-GDR & balanced"),
    Patch(color="orange", label="Low-GDR & imbalanced")
]

fig = circos.plotfig(figsize=(10, 10))
fig.axes[0].legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.05),
                   ncol=2, fontsize=9, frameon=False)

# Central label
fig.text(
    0.5, 0.5,
    "HCMV Genome\nDiversity Ratio",
    ha="center", va="center",
    fontsize=14,
    fontweight="bold",
    fontstyle="italic",
)

# Save figure
fig.savefig(output_png, dpi=300, bbox_inches="tight")
print(f"Wrote {output_png}")

# -----------------------------
# Export QC table with balance flag
# -----------------------------
gdr_df.to_csv(output_table, sep="\t", index=False)
print(f"Wrote GDR QC table with balance flag to {output_table}")
