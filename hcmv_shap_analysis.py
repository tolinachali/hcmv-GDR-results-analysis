#!/usr/bin/env python3
"""
HCMV SNP + GDR analysis pipeline with Advanced Visualization:
- Variance filtering
- PCA, t-SNE, MFA
- Random Forest classification
- SHAP interpretation
- Genomic feature overlap analysis (INCLUDING NON-CODING RNAs)
- Comprehensive visualization suite
- Automatic plot generation
- Gene-centric SHAP aggregation

Requirements:
pip install pandas scikit-learn shap matplotlib seaborn prince numpy biopython
"""
import os, warnings, re

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc

# SHAP and MFA
import shap
import prince

# BioPython for gene annotation
from Bio import SeqIO

# Set plotting style
plt.style.use('default')
sns.set_palette("colorblind")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['font.size'] = 10

# -----------------------------
# User parameters
# -----------------------------
input_file = "rf_glm_input_snp_gdr.csv"
phenotype_col = "Phenotype"
random_state = 42
variance_threshold = 0.05
min_features_to_keep = 100

# Reference genome for gene annotation
gb_file = "/home/tolina/Desktop/hcmv_manuscript_data/project_folder/results/NC_006273.gb"

rf_param_grid = {
    "n_estimators": [200, 500],
    "max_depth": [None, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt"]
}

top_n_features_to_plot = 20
save_dir = "gdr_shap_results"
os.makedirs(save_dir, exist_ok=True)


# -----------------------------
# Visualization Functions
# -----------------------------
def create_advanced_projection_plots(X_pca, X_tsne, X_mfa, y_enc, le, save_dir):
    """Create enhanced projection plots with better styling and statistics"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    plot_kwargs = dict(s=50, alpha=0.7, edgecolor='w', linewidth=0.5)

    # Get class names and colors
    class_names = le.classes_
    colors = sns.color_palette("colorblind", len(class_names))

    # PCA
    for i, label in enumerate(np.unique(y_enc)):
        axes[0].scatter(X_pca[y_enc == label, 0], X_pca[y_enc == label, 1],
                        label=class_names[label], color=colors[i], **plot_kwargs)
    axes[0].set_title("PCA Projection\n(Linear Dimensionality Reduction)", fontweight='bold')
    axes[0].set_xlabel("Principal Component 1")
    axes[0].set_ylabel("Principal Component 2")
    axes[0].grid(True, alpha=0.3)

    # t-SNE
    for i, label in enumerate(np.unique(y_enc)):
        axes[1].scatter(X_tsne[y_enc == label, 0], X_tsne[y_enc == label, 1],
                        label=class_names[label], color=colors[i], **plot_kwargs)
    axes[1].set_title("t-SNE Projection\n(Non-linear Manifold Learning)", fontweight='bold')
    axes[1].set_xlabel("t-SNE Dimension 1")
    axes[1].set_ylabel("t-SNE Dimension 2")
    axes[1].grid(True, alpha=0.3)

    # MFA
    for i, label in enumerate(np.unique(y_enc)):
        axes[2].scatter(X_mfa.iloc[y_enc == label, 0], X_mfa.iloc[y_enc == label, 1],
                        label=class_names[label], color=colors[i], **plot_kwargs)
    axes[2].set_title("MFA Projection\n(Multiple Factor Analysis)", fontweight='bold')
    axes[2].set_xlabel("MFA Dimension 1")
    axes[2].set_ylabel("MFA Dimension 2")
    axes[2].grid(True, alpha=0.3)

    # Single legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95),
               ncol=len(class_names), frameon=True, fancybox=True, shadow=True)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(os.path.join(save_dir, "combined_projection_analysis.png"))
    plt.close()
    print(f"Advanced projection plots saved")


def create_shap_summary_plots(shap_values, features, feature_names, save_dir):
    """Create comprehensive SHAP summary plots"""
    # Bar plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, features, feature_names=feature_names,
                      plot_type="bar", max_display=top_n_features_to_plot, show=False)
    plt.title("Top Features by Mean |SHAP Value|\n(Global Feature Importance)", fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "shap_importance_bar.png"))
    plt.close()

    # Beeswarm plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, features, feature_names=feature_names,
                      max_display=min(20, top_n_features_to_plot), show=False)
    plt.title("SHAP Value Distribution\n(Impact on Model Output)", fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "shap_beeswarm_plot.png"))
    plt.close()


def create_feature_centric_plots(feature_summary_df, save_dir):
    """Create plots focused on feature-level importance (including non-coding)"""
    if feature_summary_df is None or len(feature_summary_df) == 0:
        return

    # Top features by SHAP importance
    top_features = feature_summary_df.head(15)

    plt.figure(figsize=(14, 10))
    bars = plt.barh(range(len(top_features)), top_features['SHAP_Score'],
                    color='steelblue', alpha=0.8, edgecolor='black')

    plt.yticks(range(len(top_features)), top_features['Feature_Name'])
    plt.xlabel('Cumulative SHAP Importance', fontsize=12, fontweight='bold')
    plt.ylabel('Genomic Feature', fontsize=12, fontweight='bold')
    plt.title('Top Genomic Features Driving HCMV Strain Divergence\n(Ranked by Cumulative SHAP Importance)',
              fontsize=14, fontweight='bold', pad=20)

    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + width * 0.01, bar.get_y() + bar.get_height() / 2,
                 f'{width:.4f}', ha='left', va='center', fontsize=9, fontweight='bold')

    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "top_features_shap_importance.png"))
    plt.close()


def create_gene_centric_plots(gene_summary_df, save_dir):
    """Create plots focused on gene-level importance"""
    if gene_summary_df is None or len(gene_summary_df) == 0:
        return

    # Top genes by SHAP importance
    top_genes = gene_summary_df.head(15)

    plt.figure(figsize=(14, 10))
    bars = plt.barh(range(len(top_genes)), top_genes['SHAP_Score'],
                    color='steelblue', alpha=0.8, edgecolor='black')

    plt.yticks(range(len(top_genes)), top_genes['Gene'])
    plt.xlabel('Cumulative SHAP Importance', fontsize=12, fontweight='bold')
    plt.ylabel('Gene', fontsize=12, fontweight='bold')
    plt.title('Top Genes Driving HCMV Strain Divergence\n(Ranked by Cumulative SHAP Importance)',
              fontsize=14, fontweight='bold', pad=20)

    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + width * 0.01, bar.get_y() + bar.get_height() / 2,
                 f'{width:.4f}', ha='left', va='center', fontsize=9, fontweight='bold')

    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "top_genes_shap_importance.png"))
    plt.close()


def create_genomic_feature_plots(genomic_annotation_df, save_dir):
    """Create plots showing distribution of feature types"""
    if genomic_annotation_df is None or len(genomic_annotation_df) == 0:
        return

    # Feature type distribution
    feature_type_counts = genomic_annotation_df['feature_type'].value_counts()

    plt.figure(figsize=(10, 8))
    wedges, texts, autotexts = plt.pie(feature_type_counts.values,
                                       labels=feature_type_counts.index,
                                       autopct='%1.1f%%', startangle=90,
                                       colors=sns.color_palette("Set3", len(feature_type_counts)))

    plt.title('Distribution of Genomic Feature Types\nin Top Predictive Windows',
              fontsize=14, fontweight='bold', pad=20)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "feature_type_distribution.png"))
    plt.close()

    # Overlap percentage distribution
    plt.figure(figsize=(10, 6))
    plt.hist(genomic_annotation_df['overlap_percentage'], bins=20,
             alpha=0.7, color='lightblue', edgecolor='black')
    plt.xlabel('Overlap Percentage (%)', fontweight='bold')
    plt.ylabel('Frequency', fontweight='bold')
    plt.title('Distribution of Feature-Window Overlap Percentages',
              fontsize=12, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "overlap_percentage_distribution.png"))
    plt.close()


def create_model_performance_plots(y_true, y_pred, y_proba, le, save_dir):
    """Create comprehensive model performance visualizations"""
    class_names = le.classes_

    # Enhanced confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Random Forest Confusion Matrix\n(Clinical vs Lab-adapted Strain Classification)',
              fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontweight='bold')
    plt.ylabel('True Label', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "enhanced_confusion_matrix.png"))
    plt.close()

    # ROC curve (if binary classification)
    if len(class_names) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
        plt.xlabel('False Positive Rate', fontweight='bold')
        plt.ylabel('True Positive Rate', fontweight='bold')
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontweight='bold', pad=20)
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "roc_curve.png"))
        plt.close()


# -----------------------------
# Genomic Annotation Functions (FIXED TO INCLUDE NON-CODING RNAs)
# -----------------------------
def load_genbank_annotation(gb_file_path):
    """Load and parse GenBank file for genomic annotations"""
    print(f"Loading GenBank file: {gb_file_path}")
    try:
        record = SeqIO.read(gb_file_path, "genbank")
        print(f"Successfully loaded genome with {len(record.features)} features")
        return record
    except FileNotFoundError:
        print(f"ERROR: GenBank file not found at {gb_file_path}")
        print("Genomic annotation will be skipped.")
        return None
    except Exception as e:
        # print(f"ERROR loading GenBank file: {e")
        print(f"ERROR loading GenBank file: {e}")
        return None


def get_genomic_features_for_window(window_name, genbank_record):
    """Find ALL genomic features that overlap with a given window, including non-coding RNAs"""
    if genbank_record is None:
        return []

    try:
        window_match = re.search(r'window__(\d+)-(\d+)', window_name)
        if not window_match:
            return []

        win_start = int(window_match.group(1))
        win_end = int(window_match.group(2))
    except:
        return []

    overlapping_features = []

    for feature in genbank_record.features:
        if feature.type == "source":
            continue

        feat_start = int(feature.location.start)
        feat_end = int(feature.location.end)

        if not (win_end < feat_start or win_start > feat_end):
            overlap_start = max(win_start, feat_start)
            overlap_end = min(win_end, feat_end)
            overlap_bp = overlap_end - overlap_start + 1
            window_size = win_end - win_start + 1
            overlap_percentage = (overlap_bp / window_size) * 100

            strand = feature.location.strand
            strand_symbol = "+" if strand == 1 else "-" if strand == -1 else "."

            # CRITICAL FIX: Extract names for ALL feature types, not just genes
            gene_name = "Unknown"
            product_name = "Unknown"

            # Handle non-coding RNAs specifically (miRNAs, etc.)
            if feature.type == "ncRNA":
                gene_name = feature.qualifiers.get("ncRNA_class", ["ncRNA"])[0]
                product_name = feature.qualifiers.get("product", ["Unknown ncRNA"])[0]
                # For miRNAs, use the product name as the identifier
                if "miR" in product_name:
                    gene_name = product_name
            # Handle standard genes
            elif "gene" in feature.qualifiers:
                gene_name = feature.qualifiers["gene"][0]
            elif "locus_tag" in feature.qualifiers:
                gene_name = feature.qualifiers["locus_tag"][0]

            # Get product name for all feature types
            if "product" in feature.qualifiers:
                product_name = feature.qualifiers["product"][0]

            overlapping_features.append({
                'window': window_name,
                'window_start': win_start,
                'window_end': win_end,
                'feature_type': feature.type,
                'gene_name': gene_name,
                'feature_start': feat_start,
                'feature_end': feat_end,
                'strand': strand_symbol,
                'product': product_name,
                'overlap_start': overlap_start,
                'overlap_end': overlap_end,
                'overlap_bp': overlap_bp,
                'overlap_percentage': overlap_percentage
            })

    return overlapping_features


def create_genomic_annotation_report(top_features, genbank_record):
    """Create a comprehensive report of genomic features overlapping top windows"""
    all_annotations = []

    for feature_name in top_features:
        features = get_genomic_features_for_window(feature_name, genbank_record)
        all_annotations.extend(features)

    return pd.DataFrame(all_annotations)


# -----------------------------
# Main Pipeline
# -----------------------------
print("Loading dataset:", input_file)
df = pd.read_csv(input_file, index_col=0)
print("Dataset shape:", df.shape)

if phenotype_col not in df.columns:
    raise KeyError(f"Phenotype column '{phenotype_col}' not found in {input_file}")

X = df.drop(columns=[phenotype_col])
y = df[phenotype_col].astype(str)
print("Features:", X.shape, "Target:", y.shape)
print("Target distribution:\n", y.value_counts())

le = LabelEncoder()
y_enc = le.fit_transform(y)

# Load GenBank annotations
genbank_record = load_genbank_annotation(gb_file)

# Variance filtering
print(f"\nApplying VarianceThreshold (threshold={variance_threshold})")
vt = VarianceThreshold(threshold=variance_threshold)
X_reduced_arr = vt.fit_transform(X)
feature_mask = vt.get_support()
reduced_feature_names = X.columns[feature_mask].to_numpy()

if X_reduced_arr.shape[1] < min_features_to_keep:
    for t in [0.02, 0.01, 0.005, 0.001]:
        vt = VarianceThreshold(threshold=t)
        X_reduced_arr = vt.fit_transform(X)
        feature_mask = vt.get_support()
        reduced_feature_names = X.columns[feature_mask].to_numpy()
        if X_reduced_arr.shape[1] >= min_features_to_keep:
            variance_threshold = t
            print(f"Adjusted variance_threshold -> {t}, features kept: {X_reduced_arr.shape[1]}")
            break

X_reduced = pd.DataFrame(X_reduced_arr, index=X.index, columns=reduced_feature_names)
print("After variance filtering:", X_reduced.shape)

# PCA
print("\nRunning PCA...")
pca = PCA(n_components=10, random_state=random_state)
X_pca = pca.fit_transform(X_reduced)

# t-SNE
print("\nRunning t-SNE...")
tsne = TSNE(n_components=2, random_state=random_state, init='pca', learning_rate='auto')
X_tsne = tsne.fit_transform(X_reduced)

# MFA
print("\nRunning MFA...")
num_groups = 10
group_sizes = [X_reduced.shape[1] // num_groups] * num_groups
for i in range(X_reduced.shape[1] % num_groups):
    group_sizes[i] += 1

groups = []
start = 0
for g in group_sizes:
    groups.append(list(range(start, start + g)))
    start += g

group_dict = {f"group{i + 1}": X_reduced.columns[grp].tolist() for i, grp in enumerate(groups)}
X_numeric = X_reduced.apply(pd.to_numeric)

mfa = prince.MFA(n_components=2, n_iter=3, copy=True, engine='sklearn')
mfa = mfa.fit(X_numeric, groups=group_dict)
X_mfa = mfa.row_coordinates(X_numeric)

# Create projection plots
create_advanced_projection_plots(X_pca, X_tsne, X_mfa, y_enc, le, save_dir)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_reduced, y_enc, test_size=0.30, random_state=random_state, stratify=y_enc
)
print("Training set:", X_train.shape, "Test set:", X_test.shape)

# Random Forest
print("\nRunning Random Forest GridSearchCV...")
rf = RandomForestClassifier(random_state=random_state, class_weight='balanced')
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
rf_grid = GridSearchCV(rf, rf_param_grid, cv=cv, scoring="accuracy", n_jobs=-1, verbose=2)
rf_grid.fit(X_train, y_train)

best_rf = rf_grid.best_estimator_
print("Best RF params:", rf_grid.best_params_)

y_pred_rf = best_rf.predict(X_test)
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rf, target_names=le.classes_))
print("RF Accuracy:", accuracy_score(y_test, y_pred_rf))

# Model performance plots
y_pred_proba = best_rf.predict_proba(X_test)
create_model_performance_plots(y_test, y_pred_rf, y_pred_proba, le, save_dir)

# SHAP analysis
print("\nStarting SHAP analysis...")
bg_size = min(100, X_train.shape[0])
X_background = X_train.sample(n=bg_size, random_state=random_state)

explainer_rf = shap.TreeExplainer(best_rf)
shap_values_rf = explainer_rf.shap_values(X_test)

if isinstance(shap_values_rf, list):
    if len(shap_values_rf) == 2:
        shap_rf_class = shap_values_rf[1]
    else:
        shap_rf_class = np.mean(np.abs(np.array(shap_values_rf)), axis=0)
else:
    shap_rf_class = shap_values_rf

if shap_rf_class.ndim == 3:
    shap_rf_class = np.mean(np.abs(shap_rf_class), axis=2)

shap_rf_df = pd.DataFrame(shap_rf_class, index=X_test.index, columns=X_test.columns)

# SHAP plots
create_shap_summary_plots(shap_rf_df.values, X_test, X_reduced.columns, save_dir)

# Genomic annotation analysis
print("\n=== GENOMIC FEATURE ANNOTATION ===")
mean_shap_values = np.mean(np.abs(shap_rf_df.values), axis=0)
top_feature_indices = np.argsort(mean_shap_values)[::-1][:top_n_features_to_plot]
top_features = X_reduced.columns[top_feature_indices].tolist()
top_shap_scores = mean_shap_values[top_feature_indices]

print(f"Top {len(top_features)} features by SHAP importance:")
for i, (feature, score) in enumerate(zip(top_features, top_shap_scores)):
    print(f"{i + 1:2d}. {feature}: {score:.6f}")

# Genomic annotation
if genbank_record is not None:
    genomic_annotation_df = create_genomic_annotation_report(top_features, genbank_record)

    if not genomic_annotation_df.empty:
        annotation_path = os.path.join(save_dir, "genomic_annotations_detailed.csv")
        genomic_annotation_df.to_csv(annotation_path, index=False)
        print(f"Detailed genomic annotations saved to: {annotation_path}")

        # Create summary - MODIFIED TO INCLUDE NON-CODING FEATURES
        summary_data = []
        for feature in top_features:
            window_annotations = genomic_annotation_df[genomic_annotation_df['window'] == feature]
            if not window_annotations.empty:
                features_list = []
                for _, ann in window_annotations.iterrows():
                    # Include ALL features, not just those with gene_name != 'Unknown'
                    if ann['feature_type'] == 'ncRNA':
                        # Format ncRNAs specially: "hcmv-miR-UL70-5p (ncRNA+)"
                        feature_desc = f"{ann['gene_name']} ({ann['feature_type']}{ann['strand']})"
                    else:
                        feature_desc = f"{ann['gene_name']} ({ann['feature_type']}{ann['strand']})"
                    if feature_desc not in features_list:
                        features_list.append(feature_desc)

                features_list_str = ", ".join(features_list)
                num_features = len(window_annotations)
            else:
                features_list_str = "Intergenic/Unknown"
                num_features = 0

            summary_data.append({
                'Window': feature,
                'SHAP_Score': top_shap_scores[top_features.index(feature)],
                'Genomic_Features': features_list_str,
                'Num_Features': num_features
            })

        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(save_dir, "genomic_annotations_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"Genomic annotations summary saved to: {summary_path}")

        print("\n=== TOP WINDOWS WITH GENOMIC ANNOTATIONS ===")
        print(summary_df.to_string(index=False))

        # Genomic feature plots
        create_genomic_feature_plots(genomic_annotation_df, save_dir)

        # Feature-centric analysis - MODIFIED TO INCLUDE NON-CODING FEATURES
        feature_shap_data = []
        for _, row in genomic_annotation_df.iterrows():
            # Include features with meaningful names (not 'Unknown')
            if row['gene_name'] != 'Unknown' or row['feature_type'] == 'ncRNA':
                window_idx = top_features.index(row['window'])
                shap_score = top_shap_scores[window_idx]
                feature_shap_data.append({
                    'Feature_Name': row['gene_name'],
                    'Feature_Type': row['feature_type'],
                    'SHAP_Score': shap_score,
                    'Product': row['product']
                })

        if feature_shap_data:
            feature_summary_df = pd.DataFrame(feature_shap_data)
            # Group by both name and type to avoid merging different feature types
            feature_summary_df = feature_summary_df.groupby(['Feature_Name', 'Feature_Type']).agg({
                'SHAP_Score': 'sum',
                'Product': 'first'
            }).reset_index().sort_values('SHAP_Score', ascending=False)

            feature_summary_path = os.path.join(save_dir, "feature_shap_summary.csv")
            feature_summary_df.to_csv(feature_summary_path, index=False)
            print(f"Feature-centric SHAP summary saved to: {feature_summary_path}")

            print("\nTop genomic features by cumulative SHAP importance:")
            print(feature_summary_df.head(15).to_string(index=False))

            # Create feature-centric plots
            create_feature_centric_plots(feature_summary_df, save_dir)

            # --- NEW: GENE-CENTRIC SUMMARY ---
            print("\n=== GENE-CENTRIC SHAP SUMMARY (AGGREGATED) ===")
            # Create a mapping to group features by their parent gene
            gene_shap_data = []

            for _, row in genomic_annotation_df.iterrows():
                if row['gene_name'] != 'Unknown' or row['feature_type'] == 'ncRNA':
                    window_idx = top_features.index(row['window'])
                    shap_score = top_shap_scores[window_idx]

                    # For standard features, use the gene_name directly
                    # For miRNAs, they're already properly named in gene_name
                    gene_name = row['gene_name']

                    gene_shap_data.append({
                        'Gene': gene_name,
                        'Feature_Type': row['feature_type'],
                        'SHAP_Score': shap_score,
                        'Product': row['product']
                    })

            if gene_shap_data:
                # Create gene-centric summary by summing SHAP scores for each gene
                gene_summary_list = []
                gene_groups = {}

                # Group data by gene using a loop
                for item in gene_shap_data:
                    gene = item['Gene']
                    if gene not in gene_groups:
                        gene_groups[gene] = {
                            'SHAP_Score': 0,
                            'Feature_Types': set(),
                            'Product': item['Product']
                        }
                    gene_groups[gene]['SHAP_Score'] += item['SHAP_Score']
                    gene_groups[gene]['Feature_Types'].add(item['Feature_Type'])

                # Convert the grouped data to a list of dictionaries
                for gene, data in gene_groups.items():
                    # Sort and join feature types
                    sorted_types = sorted(data['Feature_Types'])
                    feature_types_str = ', '.join(sorted_types)

                    gene_summary_list.append({
                        'Gene': gene,
                        'SHAP_Score': data['SHAP_Score'],
                        'Feature_Types': feature_types_str,
                        'Product': data['Product']
                    })

                # Create DataFrame and sort
                gene_summary_df = pd.DataFrame(gene_summary_list)
                gene_summary_df = gene_summary_df.sort_values('SHAP_Score', ascending=False)

                gene_summary_path = os.path.join(save_dir, "gene_centric_shap_summary.csv")
                gene_summary_df.to_csv(gene_summary_path, index=False)
                print(f"Gene-centric SHAP summary saved to: {gene_summary_path}")

                print("\nTop genes by cumulative SHAP importance:")
                print(gene_summary_df.head(15).to_string(index=False))

                # Create gene-centric visualization
                create_gene_centric_plots(gene_summary_df, save_dir)

                # Create a combined table for discussion
                print("\n=== COMBINED VIEW FOR DISCUSSION ===")
                combined_data = []
                for _, row in gene_summary_df.iterrows():
                    if row['SHAP_Score'] > 0.001:  # Filter for significant contributors
                        combined_data.append({
                            'Gene': row['Gene'],
                            'Cumulative_SHAP_Score': row['SHAP_Score'],
                            'Feature_Types': row['Feature_Types'],
                            'Product': row['Product']
                        })

                combined_df = pd.DataFrame(combined_data)
                combined_path = os.path.join(save_dir, "combined_gene_summary.csv")
                combined_df.to_csv(combined_path, index=False)
                print(f"Combined gene summary for discussion saved to: {combined_path}")
                print(combined_df.to_string(index=False))

    else:
        print("No genomic features found overlapping with top windows.")
else:
    print("Skipping genomic annotation - no GenBank record available.")

print("\n✅ Pipeline finished successfully!")
print("All plots and results are saved in:", save_dir)