# DCM metric normalization

This repository contains the analysis scripts related to morphometric measures normalization.

The MRI data of healthy individuals from [spine-generic project](https://spine-generic.readthedocs.io) were used to 
construct a database of quantitative metrics in the PAM50 reference space; for details, see [PAM50-normalized-metrics](https://github.com/spinalcordtoolbox/PAM50-normalized-metrics).

### Dependencies

- Python 3 and packages listed in [requirements.txt](https://github.com/sct-pipeline/dcm-metric-normalization/blob/main/requirements.txt)
- [SCT](https://github.com/spinalcordtoolbox/spinalcordtoolbox/tree/master)

### Usage

Each dataset has its own processing script containing preprocessing, spinal cord segmentation and vertebral 
labeling. The analysis pipeline slightly differs between datasets based on available images.

Scripts for processing of individual datasets are located under the `scripts` folder and can be run using the 
[`sct_run_batch`](https://spinalcordtoolbox.com/user_section/command-line.html?highlight=sct_run_batch#sct-run-batch) 
wrapper script (part of the [Spinal Cord Toolbox](https://github.com/spinalcordtoolbox/spinalcordtoolbox/tree/master)). For example:

```bash
sct_run_batch -path-data PATH_TO_DATA -path-output PATH_TO_OUTPUT -script scripts/process_data_spine-generic.sh
```

or using the JSON configuration file:

```bash
sct_run_batch -c etc/config_process_data.json
```

> [!NOTE]
> The `etc/config_process_data.json` configuration file is just a template --> you need to edit it to match your data.

> [!NOTE]
> There is an `exclude.yml` file associated with each dataset under the `etc` folder. This file lists subjects/images which should be excluded from analyses. You can exclude those subjects using the `-exclude-list` flag (when running `sct_run_batch` from CLI) or the `exclude_list` key (when running `sct_run_batch` using JSON configuration file).

---

> [!TIP]
> For manual corrections of spinal cord segmentations, please refer to this repository: [manual-correction](https://github.com/spinalcordtoolbox/manual-correction) and its [wiki](https://github.com/spinalcordtoolbox/manual-correction/wiki) for examples.
