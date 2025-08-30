import os
import pandas as pd
from Bio import SeqIO
import numpy as np

# -----------------------------
# User Inputs
# -----------------------------
gdr_file = "/home/tolina/Desktop/hcmv_manuscript_data/project_folder/window/strain_type/GDRcalculationResultsJ/hcmv_gdr__dist_mat_list__Strain_Type__pvals.tsv"
metadata_file = "/home/tolina/Desktop/CMV_data/analysis_v1/all_willowj/pass_willowj/strain_.tsv"
fasta_dir = "/home/tolina/Desktop/hcmv_manuscript_data/project_folder/window/feature_alignments"
output_file = "rf_glm_input_snp_gdr.csv"

# Parameters
pval_cutoff = 0.05
low_freq_thresh = 0.05  # remove SNPs present in <5% of samples
max_gap_frac = 0.2      # remove SNPs with >20% gaps

# -----------------------------
# Load Metadata and GDR
# -----------------------------
metadata = pd.read_csv(metadata_file, sep="\t")
sample_to_type = dict(zip(metadata["Sample_ID"], metadata["Strain_Type"]))

gdr_df = pd.read_csv(gdr_file, sep="\t")
# Use only significant windows
sig_gdr_df = gdr_df[gdr_df["Strain_Type(p-value)"] <= pval_cutoff].copy()
sig_gdr_df.reset_index(drop=True, inplace=True)

print(f"Significant GDR windows: {len(sig_gdr_df)}")

# -----------------------------
# Function to generate SNP matrix for one window
# -----------------------------
def snp_matrix_from_fasta(fasta_path):
    records = list(SeqIO.parse(fasta_path, "fasta"))
    if len(records) == 0:
        return None, []

    ref_seq = str(records[0].seq)  # reference is first sequence
    samples = [rec.id for rec in records]
    snp_dict = {}

    for i in range(len(ref_seq)):
        ref_base = ref_seq[i].upper()
        snp_column = []
        for rec in records:
            base = rec.seq[i].upper()
            # SNP: 1 if different from reference, 0 if same, ignore gaps
            if base == "-":
                snp_column.append(np.nan)
            elif base != ref_base:
                snp_column.append(1)
            else:
                snp_column.append(0)
        snp_dict[i] = snp_column

    snp_df = pd.DataFrame(snp_dict, index=samples)
    return snp_df, samples

# -----------------------------
# Build combined SNP + GDR dataframe
# -----------------------------
all_snp_dfs = []
window_features = []

for idx, row in sig_gdr_df.iterrows():
    feature_name = row["Feature"].replace("_aic_distmax","")
    fasta_file = os.path.join(fasta_dir, f"{feature_name}.fasta")
    if not os.path.exists(fasta_file):
        print(f"Missing FASTA: {fasta_file}, skipping")
        continue

    snp_df, samples = snp_matrix_from_fasta(fasta_file)
    if snp_df is None:
        continue

    # Remove SNPs with too many gaps
    gap_frac = snp_df.isna().mean()
    snp_df = snp_df.loc[:, gap_frac <= max_gap_frac]

    # Remove low-frequency SNPs
    freq = snp_df.mean(skipna=True)
    snp_df = snp_df.loc[:, freq >= low_freq_thresh]

    if snp_df.shape[1] == 0:
        continue

    # Rename columns to include window
    snp_df.columns = [f"{feature_name}_pos{i}" for i in snp_df.columns]
    all_snp_dfs.append(snp_df)
    window_features.append(feature_name)

# Concatenate all SNPs
if not all_snp_dfs:
    raise RuntimeError("No SNPs passed filtering. Check thresholds or input FASTAs.")

combined_snp_df = pd.concat(all_snp_dfs, axis=1)
combined_snp_df = combined_snp_df.fillna(0).astype(int)

# Add GDR values as additional features
gdr_values = sig_gdr_df.set_index("Feature")["Strain_Type(CDR)"]
gdr_dict = {}
for wf in window_features:
    gdr_val = gdr_values.get(wf + "_aic_distmax", np.nan)
    gdr_dict[wf + "_GDR"] = [gdr_val] * combined_snp_df.shape[0]

gdr_df_final = pd.DataFrame(gdr_dict, index=combined_snp_df.index)
final_df = pd.concat([combined_snp_df, gdr_df_final], axis=1)

# Add phenotype label
final_df["Phenotype"] = [sample_to_type.get(s, "unknown") for s in final_df.index]

# Save
final_df.to_csv(output_file)
print(f"Saved RF/GLM input: {output_file}")
print(f"Columns: {list(final_df.columns)[:10]} ... Total columns: {len(final_df.columns)}")
