#!/bin/bash
#
# Run contrast-agnostic SC seg model on T2w images from DCM patients (dcm-zurich dataset)
#
# Usage:
#     run_softseg_dcm-zurich.sh <MODEL_CONFIG_FILE> <PATH_DATA>
#
# Authors: Jan Valosek, Sandrine Bedard
#

# Uncomment for full verbose
#set -x

# Immediately exit if error
set -e -o pipefail

# Exit if user presses CTRL+C (Linux) or CMD+C (OSX)
trap "echo Caught Keyboard Interrupt within script. Exiting now.; exit" INT

CONFIG_FILE=${1}
PATH_DATA=${2}
PATH_RESULTS=${CONFIG_FILE//config_file.json}     # remove config_file.json from the path

# Print retrieved variables
echo "CONFIG_FILE: ${CONFIG_FILE}"
echo "PATH_DATA: ${PATH_DATA}"
echo "PATH_RESULTS: ${PATH_RESULTS}"    # this variable is constructed from CONFIG_FILE, see above

# get starting time:
start=`date +%s`

# ------------------------------------------------------------------------------
# SCRIPT STARTS HERE
# ------------------------------------------------------------------------------
# Display useful info for the log, such as SCT version, RAM and CPU cores available
sct_check_dependencies -short

# Run contrast-agnostic SC seg model across all subjects and all contrasts (T2w ax, T2w sag)
ivadomed --segment -c "$CONFIG_FILE" --path-data "$PATH_DATA" --path-output "$PATH_RESULTS"

# Generate QC report
cd ${PATH_RESULTS}/pred_masks

# Loop across predicted files
for file_seg in sub-*.nii.gz; do
  SUBJECT=${file_seg%%_*}                    # remove everything after the first "_"
  file_image=${file_seg//"_class-0_pred"}    # remove "_class-0_pred" from file name
  sct_qc -i ${PATH_DATA}/${SUBJECT}/anat/${file_image} -s ${file_seg} -p sct_deepseg_sc -qc ${PATH_RESULTS}/qc -qc-subject ${SUBJECT}
done

# ------------------------------------------------------------------------------
# End
# ------------------------------------------------------------------------------

# Display results (to easily compare integrity across SCT versions)
end=`date +%s`
runtime=$((end-start))
echo
echo "~~~"
echo "SCT version: `sct_version`"
echo "Ran on:      `uname -nsr`"
echo "Duration:    $(($runtime / 3600))hrs $((($runtime / 60) % 60))min $(($runtime % 60))sec"
echo "~~~"
