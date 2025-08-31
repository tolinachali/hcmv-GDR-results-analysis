## GDR Calculation and Circos Visualization
This pipeline uses results from WillowJ GDR calculation to combine a visual and quantitative assessment in one.
**Inputs:**
gdr_file: TSV of GDR, associated p-values, and other statistics from the determination on a windowed basis.

gb_file: GenBank file of the reference HCMV genome used to determine genes with a track orientation.

fasta_dir: Folder of FASTA alignments on a windowed basis.
strain_ma: Metadata assigning sample IDs mapped to phenotype (Clinical/Lab).
**Windows and Significance Processing:**
	
 ***Window Coordinates:*** If the input file does not specify start or end windows for the genomic windows, they are calculated based on the genome's length.
	
 -Balanced Sample Calculation: For every window, the program opens the windowed FASTA alignment to read how many clinical (n_c) and lab (n_l) samples are present. 
 
 The window is characterized as "balanced" if the min(n_c, n_l) >= (n_c + n_l) * 0.5. This is QC as a GDR, determined by a window with one lab sample, which is not statistically valid.
 
 Directional Filtering: Windows are filtered via p-values using the directional cutoff (pval_cutoff = 0.05).
**Results Using Circos Plot:**
=>Tracks: The program generates a circular track plot of the HCMV genome.
	
 - Axis Track: Indicates kilobases of the genome.
	
 - Gene Track: Shows the genes based on annotation from the GenBank-filed input.
	
 - GDR Tracks: Three overlapping bar tracks show scaled results for:
   - GDR between groups overall (grey)

   - Intra-diversity for clinical groups (steel blue)

   - Intra-diversity for lab groups (tomato red)
	
 - Highlight Track: Windows characterized by an overall GDR lower than (<= gdr_cutoff = 0.8) are highlighted. 


**Feature Engineering for Machine Learning (**`generate_variant_on_significat_GDR_windows.py`**)**

This pipeline bridges the gap between phylogenetic analysis and machine learning by constructing a feature matrix from the significant genomic windows.

**Methodology**

1. **Inputs:** The same GDR file and metadata.

2. **SNP Matrix Construction:**

   - **Significant Windows Only:** Focuses only on windows with a significant GDR p-value (`pval_cutoff = 0.05`).

   - **SNP Calling:** For each window's FASTA alignment:

     - The first sequence is treated as the reference.
     - If we used the Merlin reference, every difference from that reference would be called a variant (1), even if that variant is actually the major allele in the population.
     - This would create artificial patterns where most samples have "1"s at positions where they differ from Merlin but are actually identical to each other.

     - For every position, a `0` is assigned if a sample matches the reference, and a `1` if it contains a variant.

     - Gaps are treated as `NaN`.

   - **Filtering:** Applies quality filters to remove uninformative SNPs:

     - `max_gap_frac = 0.2`: Removes positions with &gt;20% gaps.

     - `low_freq_thresh = 0.05`: Removes rare variants present in &lt;5% of samples. This is critical for model stability.

3. **Feature Integration:**

   - **SNP Features:** The filtered SNP positions are renamed to include their window of origin (e.g., `window_1000-1500_pos45`).

   - **GDR as a Feature:** The overall GDR value for each significant window is added as a separate feature for each sample (e.g., `window_1000-1500_GDR`). This allows the model to learn the global importance of each window.

4. **Output:**

   - `rf_glm_input_snp_gdr.csv`: A final matrix where rows are samples, columns are features (SNPs + GDR values), and the last column is the target phenotype.

**Achievement**

This pipeline successfully translates biological sequences into a structured, high-dimensional dataset suitable for statistical learning. The innovative inclusion of GDR values as features empowers the model to leverage the phylogenetic signal during prediction.
