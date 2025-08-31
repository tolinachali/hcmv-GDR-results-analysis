## GDR Calculation and Circos Visualization 
This pipeline uses results from WillowJ GDR calculation to combine a visual and quantitative assessment in one.
### Inputs:
gdr_file: TSV of GDR, associated p-values, and other statistics from the determination on a windowed basis.

gb_file: GenBank file of the reference HCMV genome used to determine genes with a track orientation.

fasta_dir: Folder of FASTA alignments on a windowed basis.
strain_ma: Metadata assigning sample IDs mapped to phenotype (Clinical/Lab).
### Windows and Significance Processing:
	
 Window Coordinates: If the input file does not specify start or end windows for the genomic windows, they are calculated based on the genome's length.
	
 -Balanced Sample Calculation: For every window, the program opens the windowed FASTA alignment to read how many clinical (n_c) and lab (n_l) samples are present. 
 
 The window is characterized as "balanced" if the min(n_c, n_l) >= (n_c + n_l) * 0.5. This is QC as a GDR, determined by a window with one lab sample, which is not statistically valid.
 
 Directional Filtering: Windows are filtered via p-values using the directional cutoff (pval_cutoff = 0.05).
### Results Using Circos Plot:
=>Tracks: The program generates a circular track plot of the HCMV genome.
	
 - Axis Track: Indicates kilobases of the genome.
	
 - Gene Track: Shows the genes based on annotation from the GenBank-filed input.
	
 - GDR Tracks: Three overlapping bar tracks show scaled results for:
		- GDR between groups overall (grey)

   - Intra-diversity for clinical groups (steel blue)

   - Intra-diversity for lab groups (tomato red)
	
 - Highlight Track: Windows characterized by an overall GDR lower than (<= gdr_cutoff = 0.8) are highlighted. 


