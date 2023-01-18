# DCM metric normalization

This repository contains the analysis scripts related to quantitative metrics normalization.

The MRI data of healthy individuals from [spine-generic project](https://spine-generic.readthedocs.io) is used to 
construct a database of quantitative metrics in the [PAM50 reference space](https://pubmed.ncbi.nlm.nih.gov/29061527/).

### Usage

Each dataset has its own processing script containing preprocessing, spinal cord segmentation and vertebral 
labeling. The analysis pipeline slightly differs between datasets based on available images.

Scripts for processing of individual datasets are located in the `scripts` folder and can be run using the 
[`sct_run_batch`](https://spinalcordtoolbox.com/user_section/command-line.html?highlight=sct_run_batch#sct-run-batch) 
wrapper script (part of the [Spinal Cord Toolbox](https://github.com/spinalcordtoolbox/spinalcordtoolbox/tree/master)). For example:

```bash
sct_run_batch -path-data PATH_TO_DATA -path-output PATH_TO_OUTPUT -script scripts/process_data_spine-generic.sh
```

or using the configuration file:

```bash
sct_run_batch -c etc/config_process_data.json
```

> **Note** The `etc/config_process_data.json` configuration file is just a template --> you need to edit it to match your data.

### Dependencies

[SCT](https://github.com/spinalcordtoolbox/spinalcordtoolbox/tree/master)