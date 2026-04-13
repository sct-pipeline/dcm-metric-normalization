#!/bin/bash
#
# Process spine-generic multi-subject dataset: compute anterior and posterior lengths from T2w isotropic images.
# https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/5199
#
# For each subject, the script:
#   1. Segments the spinal cord (uses manual segmentation from derivatives/labels if available,
#      otherwise runs sct_deepseg spinalcord automatically)
#   2. Labels intervertebral discs (uses manual labels from derivatives/labels if available,
#      otherwise runs TotalSpineSeg automatically)
#   3. Computes per-slice cord morphometrics (including length_anterior and length_posterior)
#      and appends results to T2w_cord_metrics_perlevel.csv
#
# Dataset: https://github.com/spine-generic/data-multi-subject
#
# Usage:
#     sct_run_batch -c etc/config_process_data_spine-generic_ap_lengths.json
#
# Example JSON configuration file:
#   {
#     "path_data"   : "~/data/data.neuro.polymtl.ca/data-multi-subject",
#     "path_output" : "~/results/spine-generic/spine-generic_ap_lengths_2026-04-10",
#     "script"      : "~/code/dcm-metric-normalization/scripts/process_data_spine-generic_ap_lengths.sh",
#     "jobs"        : 8
#   }
#
# The following global variables are retrieved from the caller sct_run_batch
# but could be overwritten by uncommenting the lines below:
# PATH_DATA_PROCESSED="~/data_processed"
# PATH_RESULTS="~/results"
# PATH_LOG="~/log"
# PATH_QC="~/qc"
#
# Authors: Jan Valosek
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

# CONVENIENCE FUNCTIONS
# ======================================================================================================================
# Check if manual spinal cord segmentation file already exists. If it does, copy it locally.
# If it doesn't, perform automatic spinal cord segmentation.
segment_if_does_not_exist() {
  local file="$1"
  local contrast="$2"   # only for logging
  # Update global variable with segmentation file name
  FILESEG="${file}_label-SC_seg"
  FILESEGMANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${FILESEG}.nii.gz"
  echo
  echo "Looking for manual segmentation: $FILESEGMANUAL"
  if [[ -e $FILESEGMANUAL ]]; then
    echo "✅ [$(date '+%Y-%m-%d %H:%M:%S')] Found! Using manual spinal cord segmentation."
    echo "✅ [$(date '+%Y-%m-%d %H:%M:%S')] ${FILESEG}.nii.gz found under derivatives/labels --> using manual spinal cord segmentation" >> "${PATH_LOG}/${contrast}_SC_segmentations.log"
    rsync -avzh $FILESEGMANUAL ${FILESEG}.nii.gz
    sct_qc -i ${file}.nii.gz -s ${FILESEG}.nii.gz -p sct_deepseg_sc -qc ${PATH_QC} -qc-subject ${SUBJECT}
  else
    echo "🤖 [$(date '+%Y-%m-%d %H:%M:%S')] Not found. Proceeding with automatic spinal cord segmentation."
    echo "🤖 [$(date '+%Y-%m-%d %H:%M:%S')] ${FILESEG}.nii.gz NOT found --> segmenting spinal cord automatically" >> "${PATH_LOG}/${contrast}_SC_segmentations.log"
    sct_deepseg spinalcord -i ${file}.nii.gz -o ${FILESEG}.nii.gz -qc ${PATH_QC} -qc-subject ${SUBJECT}
  fi
}

label_if_does_not_exist() {
  local file="$1"
  # Copy manual disc labels from derivatives/labels if they exist
  FILELABEL="${file}_label-discs_dlabel"
  FILELABELMANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${FILELABEL}.nii.gz"
  echo "Looking for manual disc labels: $FILELABELMANUAL"
  if [[ -e $FILELABELMANUAL ]]; then
    echo "✅ [$(date '+%Y-%m-%d %H:%M:%S')] Found! Using manual disc labels."
    echo "✅ [$(date '+%Y-%m-%d %H:%M:%S')] ${FILELABEL}.nii.gz found under derivatives/labels --> using manual disc labels" >> "${PATH_LOG}/T2w_disc_labels.log"
    rsync -avzh $FILELABELMANUAL ${FILELABEL}.nii.gz
  else
    echo "🤖 [$(date '+%Y-%m-%d %H:%M:%S')] Manual disc labels not found. Proceeding with automatic labeling using TotalSpineSeg."
    echo "🤖 [$(date '+%Y-%m-%d %H:%M:%S')] ${FILELABEL}.nii.gz NOT found --> using automatic labeling using TotalSpineSeg" >> "${PATH_LOG}/T2w_disc_labels.log"
    sct_deepseg totalspineseg -i ${file}.nii.gz -step1-only 1
    # Keep only disc labels and remove other outputs
    mv ${file}_step1_levels.nii.gz ${FILELABEL}.nii.gz
    rm ${file}_step1_cord.nii.gz ${file}_step1_canal.nii.gz ${file}_step1_output.nii.gz
    # Create disc labels QC
    sct_qc -i ${file}.nii.gz -s ${FILELABEL}.nii.gz -p sct_label_utils -qc ${PATH_QC} -qc-subject "disc_labels_totalspineseg"
  fi
}

# Retrieve subject from sct_run_batch
SUBJECT=$1

echo "Processing ${SUBJECT}"

# get starting time:
start=`date +%s`

# ------------------------------------------------------------------------------
# SCRIPT STARTS HERE
# ------------------------------------------------------------------------------
# Display useful info for the log, such as SCT version, RAM and CPU cores available
sct_check_dependencies -short

# Go to folder where data will be copied and processed
cd $PATH_DATA_PROCESSED

# Copy participants.tsv file to the root folder of the dataset
if [[ ! -e ../participants.tsv ]]; then
    rsync -avzh ${PATH_DATA}/participants.tsv ${PATH_DATA_PROCESSED}/../participants.tsv
fi

# Copy source T2w images
rsync -Ravzh ${PATH_DATA}/./${SUBJECT}/anat/${SUBJECT}_T2w.* .

# Go to subject anat folder
cd ${SUBJECT}/anat

# ------------------------------------------------------------------------------
# T2w iso
# ------------------------------------------------------------------------------
file_t2w="${SUBJECT}_T2w"

if [[ ! -e ${file_t2w}.nii.gz ]]; then
    echo "File ${file_t2w}.nii.gz does not exist" >> ${PATH_LOG}/missing_files.log
    echo "ERROR: File ${file_t2w}.nii.gz does not exist. Exiting."
    exit 1
fi

# Segment SC (if SC segmentation file already exists under derivatives folder, it will be copied)
segment_if_does_not_exist ${file_t2w} 'T2w'
file_t2w_seg=$FILESEG

# Label discs (if disc labels already exist under derivatives folder, they will be copied)
label_if_does_not_exist ${file_t2w}
file_t2w_labels=${file_t2w}_label-discs_dlabel

# -------------
# Compute length_anterior and length_posterior
# https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/5199
# -------------
echo "Computing spinal cord morphometrics..."
# Compute cord metrics perslice in the native space -- metrics across subjects are appended to a single CSV file
# Note: -anat is used for QC purposes to display the metrics on the original image space.
sct_process_segmentation -anat ${file_t2w}.nii.gz -i ${file_t2w_seg}.nii.gz -discfile ${file_t2w_labels}.nii.gz -perslice 1 -o ${PATH_RESULTS}/T2w_cord_metrics_perslice.csv -append 1 -qc ${PATH_QC}

# Compute cord metrics in the PAM50 ('-normalize-PAM50' flag)
# Note: '-v 2' flag is used to get all available vertebral levels from PAM50 template. This assures that the output CSV
# files will have the same number of rows, regardless of the subject's vertebral levels.
mkdir -p ${PATH_RESULTS}/PAM50
sct_process_segmentation -i ${file_t2w_seg}.nii.gz -discfile ${file_t2w_labels}.nii.gz -perslice 1 -normalize-PAM50 1 -v 2 -o ${PATH_RESULTS}/PAM50/${file_t2w}_PAM50.csv


echo "Finished processing ${file_t2w}" >> ${PATH_LOG}/processed_files_T2w.log

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