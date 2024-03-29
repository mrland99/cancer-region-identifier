# spatial-research
Repository to store the code for my research conducted as part of the Raphael Lab at Princeton University.

Data needed to run code can be found at: https://www.spatialresearch.org/resources-published-datasets/doi-10-1126science-aaf2403/.

You will need:
* Count matrices of the four layers
* Images of the four layers

## Abstract:

Previous methods of determining normal cells vs cancer cells based on gene
expression data have relied on analyzing a subset of tissue specific genes (e.g.
BRCA2 mutation in breast tissue). However, the diversity of gene expression in
cancer cells presents difficulties for this approach. In this paper, we present two
novel classification methods that are not dependent on the presence of specific genes.
Instead, these methods consider the entire transcriptome as a whole and use PCA
and percent dropout to reduce our data to one dimensional feature vectors. Next,
we apply a simple thresholding classification method to these feature vectors to
distinguish normal cells from healthy cells. We find that this novel clustering
approach according to the first principal component of PCA and percent dropout
have high performance and present a promising new approach in identifying tumor
regions in cancer tissue.
