# DCM metric normalization

This repository contains the analysis scripts related to quantitative metrics normalization.

The MRI data of healthy individuals from [spine-generic project](https://spine-generic.readthedocs.io) is used to 
construct a database of quantitative metrics in the [PAM50 reference space](https://pubmed.ncbi.nlm.nih.gov/29061527/).

Each dataset has its own processing script containing preprocessing (optional), spinal cord segmentation and vertebral 
labeling. The analysis pipeline slightly differs between datasets based on available images.

### Dependencies

[SCT](https://github.com/spinalcordtoolbox/spinalcordtoolbox/tree/master)