#!/bin/bash
#
# Compute AP and RL diameters using "traditional" and HOG-based methods
#
# Requirements:
#   SCT PR #4958: https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4958
#
# Usage:
#     xargs -n2 sh -c 'sct_run_batch -path-data $1 -path-output $2 -config config_compute_diameters.json' sh < datasets.txt
#
# Example of config.json:
# {
#  "script"      : "~/code/dcm-metric-normalization/hog_angle/compute_diameters.sh",
#  "jobs"        : 8
# }
#
# Example of dataset.txt:
#    ~/data/data.neuro.polymtl.ca/data-multi-subject/ ~/results/hog_angle/compute_diameters_2025-07-29
#    ~/data/data.neuro.polymtl.ca/whole-spine ~/results/hog_angle/compute_diameters_2025-07-29
#
# NOTE: we use the same results folder for all datasets to store results from all datasets in a single folder.
#
# Manual segmentations and disc labels should be located under:
#   PATH_DATA/derivatives/labels/SUBJECT/<CONTRAST>/
#
# Author: Jan Valosek
#

# Uncomment for full verbose
set -x

# Immediately exit if error
set -e -o pipefail

# Exit if user presses CTRL+C (Linux) or CMD+C (OSX)
trap "echo Caught Keyboard Interrupt within script. Exiting now.; exit" INT

# Print retrieved variables from the sct_run_batch script to the log (to allow easier debug)
echo "Retrieved variables from from the caller sct_run_batch:"
echo "PATH_DATA: ${PATH_DATA}"
echo "PATH_DATA_PROCESSED: ${PATH_DATA_PROCESSED}"
echo "PATH_RESULTS: ${PATH_RESULTS}"
echo "PATH_LOG: ${PATH_LOG}"
echo "PATH_QC: ${PATH_QC}"

SUBJECT=$1

echo "SUBJECT: ${SUBJECT}"

# get starting time:
start=`date +%s`

# ------------------------------------------------------------------------------
# SCRIPT STARTS HERE
# ------------------------------------------------------------------------------
# Display useful info for the log, such as SCT version, RAM and CPU cores available
sct_check_dependencies -short

cd $PATH_DATA/$SUBJECT/anat
git annex get *T2w*
cd $PATH_DATA/derivatives/labels/$SUBJECT/anat
git annex get *T2w*

# Go to folder where data will be copied and processed
cd $PATH_DATA_PROCESSED

# Copy source images
# Note: we use '/./' in order to include the sub-folder 'ses-0X'
rsync -avzh $PATH_DATA/./$SUBJECT .
#	•	-a → Archive mode (preserves most attributes except group ownership).
#	•	-v → Verbose output.
#	•	-z → Compresses data during transfer.
#	•	-h → Human-readable file sizes.
#	•	--no-g → Skips changing group ownership.

# Remove dwi folder if it exists (because we do not process DWI data in this script)
if [[ -d ${SUBJECT}/dwi ]]; then
  echo "Removing DWI folder: ${SUBJECT}/dwi"
  rm -rf ${SUBJECT}/dwi
fi

# Go to subject folder for source images
cd ${SUBJECT}/anat

# Define variables
# We do a substitution '/' --> '_' in case there is a subfolder 'ses-0X/'
file="${SUBJECT//[\/]/_}"
echo "file: ${file}"

# -------------------------------------------------------------------------
# T2w
# -------------------------------------------------------------------------
# Steps:
#   - find T2w image
#   - remove all other files except the selected T2w and its derivatives to save space
#   - copy SC segmentation from derivatives/labels folder (we assume it exists)
#   - copy disc labels from derivatives/labels folder (we assume it exists)
#   - generate labeled segmentation using init disc labels
#   - generate metrics using sct_process_segmentation
#   - generate hog_angle using custom script

# Find T2w image
if [[ -e "${file}_T2w.nii.gz" ]]; then
  file_t2="${file}_T2w"
else
  echo "No suitable T2w file found for subject ${SUBJECT}" >&2
  exit 1
fi

# Remove all other files except the selected T2w and its derivatives to save space
find . -maxdepth 1 -type f \
  ! -name "${file_t2}.*" \
  ! -name "${file_t2}_*" \
  -exec rm -f {} +
echo "file_t2: ${file_t2}"

echo "👉 Processing: ${file_t2}"

# Copy SC segmentation from derivatives/labels folder (we assume it exists)
FILESEG="${file_t2}_label-SC_seg"
rsync -avzh --no-g "${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${FILESEG}.nii.gz" "${FILESEG}.nii.gz"
echo "✅ [$(date '+%Y-%m-%d %H:%M:%S')] ${FILESEG}.nii.gz found under derivatives/labels --> copying it" >> "${PATH_LOG}/T2w_sc_segmentation.log"

# Copy disc labels from derivatives/labels folder (we assume it exists) as we need them for vertebral labeling
FILEDISCS="${file_t2}_labels-disc-manual"
# whole-spine
if [[ -e "${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${file_t2}_labels-disc-manual.nii.gz" ]]; then
    rsync -avzh --no-g "${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${file_t2}_labels-disc-manual.nii.gz" "${FILEDISCS}.nii.gz"
    echo "✅ [$(date '+%Y-%m-%d %H:%M:%S')] ${file_t2}_labels-disc-manual.nii.gz found under derivatives/labels --> copying it" >> "${PATH_LOG}/T2w_disc_labels.log"
# spine-generic
elif [[ -e "${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${file_t2}_label-discs_dlabel.nii.gz" ]]; then
    rsync -avzh --no-g "${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${file_t2}_label-discs_dlabel.nii.gz" "${FILEDISCS}.nii.gz"
    echo "✅ [$(date '+%Y-%m-%d %H:%M:%S')] ${file_t2}_label-discs_dlabel.nii.gz found under derivatives/labels --> copying it" >> "${PATH_LOG}/T2w_disc_labels.log"
fi

# Generate labeled segmentation using init disc labels
sct_label_vertebrae -i ${file_t2}.nii.gz -s ${FILESEG}.nii.gz -discfile ${FILEDISCS}.nii.gz -c t2 -qc ${PATH_QC} -qc-subject ${file}

# Normalize to PAM50, with and without the angle correction
sct_process_segmentation -i ${file_t2}.nii.gz -s ${FILESEG}.nii.gz -vertfile ${FILESEG}_labeled.nii.gz -perslice 1 -normalize-PAM50 1 -o ${PATH_RESULTS}/${file}_metrics_PAM50_angle_corr0.csv -qc ${PATH_QC} -angle-corr 0 -v 2
sct_process_segmentation -i ${file_t2}.nii.gz -s ${FILESEG}.nii.gz -vertfile ${FILESEG}_labeled.nii.gz -perslice 1 -normalize-PAM50 1 -o ${PATH_RESULTS}/${file}_metrics_PAM50_angle_corr1.csv -qc ${PATH_QC} -angle-corr 1

# No normalization to PAM50, with and without the angle correction
sct_process_segmentation -i ${file_t2}.nii.gz -s ${FILESEG}.nii.gz -vertfile ${FILESEG}_labeled.nii.gz -perslice 1 -o ${PATH_RESULTS}/${file}_metrics_angle_corr0.csv -qc ${PATH_QC} -angle-corr 0
sct_process_segmentation -i ${file_t2}.nii.gz -s ${FILESEG}.nii.gz -vertfile ${FILESEG}_labeled.nii.gz -perslice 1 -o ${PATH_RESULTS}/${file}_metrics_angle_corr1.csv -qc ${PATH_QC} -angle-corr 1

# Create figures folder under ${PATH_RESULTS} if it does not exist
mkdir -p ${PATH_RESULTS}/figures

# Generate figure using custom script (assuming this repo is cloned in `~/code/dcm-metric-normalization`)
${SCT_DIR}/python/envs/venv_sct/bin/python ~/code/dcm-metric-normalization/hog_angle/AP_RL_diameters_PAM50.py \
  -i ${PATH_RESULTS}/${file}_metrics_PAM50_angle_corr0.csv \
  -o ${PATH_RESULTS}/figures/${file}_AP_RL_diameters_PAM50_angle_corr0.png \
  -smooth 0

${SCT_DIR}/python/envs/venv_sct/bin/python ~/code/dcm-metric-normalization/hog_angle/AP_RL_diameters_PAM50.py \
  -i ${PATH_RESULTS}/${file}_metrics_PAM50_angle_corr1.csv \
  -o ${PATH_RESULTS}/figures/${file}_AP_RL_diameters_PAM50_angle_corr1.png \
  -smooth 0

${SCT_DIR}/python/envs/venv_sct/bin/python ~/code/dcm-metric-normalization/hog_angle/AP_RL_diameters.py \
  -i ${PATH_RESULTS}/${file}_metrics_angle_corr0.csv \
  -o ${PATH_RESULTS}/figures/${file}_AP_RL_diameters_angle_corr0.png \

${SCT_DIR}/python/envs/venv_sct/bin/python ~/code/dcm-metric-normalization/hog_angle/AP_RL_diameters.py \
  -i ${PATH_RESULTS}/${file}_metrics_angle_corr1.csv \
  -o ${PATH_RESULTS}/figures/${file}_AP_RL_diameters_angle_corr1.png \

# ------------------------------------------------------------------------------
# End
# ------------------------------------------------------------------------------
# Display useful info for the log
end=`date +%s`
runtime=$((end-start))
echo
echo "~~~"
echo "SCT version: `sct_version`"
echo "Ran on:      `uname -nsr`"
echo "Duration:    $(($runtime / 3600))hrs $((($runtime / 60) % 60))min $(($runtime % 60))sec"
echo "~~~"
