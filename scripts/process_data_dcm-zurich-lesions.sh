#!/bin/bash
#
# Run `sct_compute_compression` on T2w images from DCM patients (dcm-zurich-lesions dataset)
#
# Usage:
#     sct_run_batch -c <PATH_TO_REPO>/etc/config_process_data.json
#
# The following global variables are retrieved from the caller sct_run_batch
# but could be overwritten by uncommenting the lines below:
# PATH_DATA_PROCESSED="~/data_processed"
# PATH_RESULTS="~/results"
# PATH_LOG="~/log"
# PATH_QC="~/qc"
#
# Authors: Jan Valosek, Sandrine Bedard, Julien Cohen-Adad
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
# If it doesn't, perform automatic spinal cord segmentation
segment_if_does_not_exist() {
  local file="$1"
  local contrast="$2"
  # Update global variable with segmentation file name
  FILESEG="${file}_label-SC_mask"
  # TODO - change to FILESEG="${file}_label-SC_seg"
  FILESEGMANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${FILESEG}-manual.nii.gz"
  echo
  echo "Looking for manual segmentation: $FILESEGMANUAL"
  if [[ -e $FILESEGMANUAL ]]; then
    echo "Found! Using manual segmentation."
    rsync -avzh $FILESEGMANUAL ${FILESEG}.nii.gz
    sct_qc -i ${file}.nii.gz -s ${FILESEG}.nii.gz -p sct_deepseg_sc -qc ${PATH_QC} -qc-subject ${SUBJECT}
  else
    echo "Not found. Proceeding with automatic segmentation."
    # Segment spinal cord
    sct_deepseg_sc -i ${file}.nii.gz -o ${FILESEG}.nii.gz -c ${contrast} -qc ${PATH_QC} -qc-subject ${SUBJECT}
  fi
}

# Check if manual label already exists. If it does, generate labeled segmentation from manual disc labels.
# If it doesn't, perform automatic spinal cord labeling
label_if_does_not_exist(){
  local file="$1"
  local file_seg="$2"
  local contrast="$3"
  # Update global variable with segmentation file name
  FILELABEL="${file}_label-disc"
  FILELABELMANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${FILELABEL}-manual.nii.gz"
  echo "Looking for manual disc labels: $FILELABELMANUAL"
  if [[ -e $FILELABELMANUAL ]]; then
    echo "Found! Using manual disc labels."
    rsync -avzh $FILELABELMANUAL ${FILELABEL}.nii.gz
    # Generate labeled segmentation from manual disc labels
    sct_label_vertebrae -i ${file}.nii.gz -s ${file_seg}.nii.gz -discfile ${FILELABEL}.nii.gz -c ${contrast} -qc ${PATH_QC} -qc-subject ${SUBJECT}
  else
    echo "Not found. Proceeding with automatic labeling."
    # Generate labeled segmentation automatically (no manual disc labels provided)
    sct_label_vertebrae -i ${file}.nii.gz -s ${file_seg}.nii.gz -c ${contrast} -qc ${PATH_QC} -qc-subject ${SUBJECT}
  fi
  # Generate QC to access disc labels created by sct_label_vertebrae
  sct_qc -i ${file}.nii.gz -s ${file_seg}_labeled_discs.nii.gz -p sct_label_utils -qc ${PATH_QC} -qc-subject ${SUBJECT}
}

# Retrieve input params and other params
SUBJECT=$1

# get starting time:
start=`date +%s`

# ------------------------------------------------------------------------------
# SCRIPT STARTS HERE
# ------------------------------------------------------------------------------
# Display useful info for the log, such as SCT version, RAM and CPU cores available
sct_check_dependencies -short

# Go to folder where data will be copied and processed
cd $PATH_DATA_PROCESSED

# Copy source T2w images
# Note: we use '/./' in order to include the sub-folder 'ses-0X'
rsync -Ravzh ${PATH_DATA}/./${SUBJECT}/anat/${SUBJECT}_*T2w.* .

# Go to subject folder for source images
cd ${SUBJECT}/anat


# ------------------------------------------------------------------------------
# T2w Axial
# ------------------------------------------------------------------------------
# Define variables
# We do a substitution '/' --> '_' in case there is a subfolder 'ses-0X/'
file_t2_ax="${SUBJECT//[\/]/_}"_acq-ax_T2w

# Segment SC (if SC segmentation file already exists under derivatives folder, it will be copied)
segment_if_does_not_exist ${file_t2_ax} 't2'
file_t2_ax_seg=$FILESEG

# Label SC
file_t2_ax_labels="${file_t2_ax}_label-disc"
FILELABELMANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${file_t2_ax_labels}-manual.nii.gz"
if [[ ! -e $FILELABELMANUAL ]]; then
    echo "File ${FILELABELMANUAL}.nii.gz does not exist" >> ${PATH_LOG}/missing_files.log
    echo "ERROR: File ${FILELABELMANUAL} does not exist. Exiting."
    exit 1
else
    echo "Found! Using manual disc labels."
    rsync -avzh $FILELABELMANUAL ${file_t2_ax_labels}.nii.gz

    # Label T2w axial spinal cord segmentation using manual disc labels.
    # Note: we use sct_label_utils instead of sct_label_vertebrae to avoid SC straightening
    # Context: https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4072
    sct_label_utils -i ${file_t2_ax_seg}.nii.gz -disc ${file_t2_ax_labels}.nii.gz -o ${file_t2_ax_seg}_labeled.nii.gz
    # Generate QC report to assess labeled segmentation
    sct_qc -i ${file_t2_ax}.nii.gz -s ${file_t2_ax_seg}_labeled.nii.gz -p sct_label_vertebrae -qc ${PATH_QC} -qc-subject ${SUBJECT}
fi

# Compute statistics on segmented lesion
file_t2_ax_lesion="${file_t2_ax}_label-lesion"
rsync -avzh "${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${file_t2_ax_lesion}.nii.gz" ${file_t2_ax_lesion}.nii.gz
sct_analyze_lesion -m ${file_t2_ax_lesion}.nii.gz -s ${file_t2_ax_seg}.nii.gz

# Check if compression labels exists
file_compression="${file_t2_ax}_label-compression"
FILE_COMPRESSION_MANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${file_compression}.nii.gz"
if [[ ! -e ${FILE_COMPRESSION_MANUAL} ]]; then
    echo "File ${FILE_COMPRESSION_MANUAL}.nii.gz does not exist" >> ${PATH_LOG}/missing_files.log
    echo "ERROR: File ${FILE_COMPRESSION_MANUAL}.nii.gz does not exist. Exiting."
    exit 1
else
    echo "Found! Using manual compression labels."
    rsync -avzh $FILE_COMPRESSION_MANUAL ${file_compression}.nii.gz

    # Compute compression morphometrics for diameter_AP with and without normalization to PAM50
    sct_compute_compression -i ${file_t2_ax_seg}.nii.gz -vertfile ${file_t2_ax_seg}_labeled.nii.gz -l ${file_compression}.nii.gz -normalize 1 -o ${PATH_RESULTS}/${file_t2_ax}_diameter_AP_norm.csv
    sct_compute_compression -i ${file_t2_ax_seg}.nii.gz -vertfile ${file_t2_ax_seg}_labeled.nii.gz -l ${file_compression}.nii.gz -normalize 0 -o ${PATH_RESULTS}/${file_t2_ax}_diameter_AP.csv

    # Compute compression morphometrics for cross-sectional area with and without normalization
    sct_compute_compression -i ${file_t2_ax_seg}.nii.gz -vertfile ${file_t2_ax_seg}_labeled.nii.gz -l ${file_compression}.nii.gz -normalize 1 -metric area -o ${PATH_RESULTS}/${file_t2_ax}_area_norm.csv
    sct_compute_compression -i ${file_t2_ax_seg}.nii.gz -vertfile ${file_t2_ax_seg}_labeled.nii.gz -l ${file_compression}.nii.gz -normalize 0 -metric area -o ${PATH_RESULTS}/${file_t2_ax}_area.csv

    # Compute compression morphometrics for diameter_RL with and without normalization
    sct_compute_compression -i ${file_t2_ax_seg}.nii.gz -vertfile ${file_t2_ax_seg}_labeled.nii.gz -l ${file_compression}.nii.gz -normalize 1 -metric diameter_RL -o ${PATH_RESULTS}/${file_t2_ax}_diameter_RL_norm.csv
    sct_compute_compression -i ${file_t2_ax_seg}.nii.gz -vertfile ${file_t2_ax_seg}_labeled.nii.gz -l ${file_compression}.nii.gz -normalize 0 -metric diameter_RL -o ${PATH_RESULTS}/${file_t2_ax}_diameter_RL.csv

    # Compute compression morphometrics for eccentricity with and without normalization
    sct_compute_compression -i ${file_t2_ax_seg}.nii.gz -vertfile ${file_t2_ax_seg}_labeled.nii.gz -l ${file_compression}.nii.gz -normalize 1 -metric eccentricity -o ${PATH_RESULTS}/${file_t2_ax}_eccentricity_norm.csv
    sct_compute_compression -i ${file_t2_ax_seg}.nii.gz -vertfile ${file_t2_ax_seg}_labeled.nii.gz -l ${file_compression}.nii.gz -normalize 0 -metric eccentricity -o ${PATH_RESULTS}/${file_t2_ax}_eccentricity.csv

    # Compute compression morphometrics for solidity with and without normalization
    sct_compute_compression -i ${file_t2_ax_seg}.nii.gz -vertfile ${file_t2_ax_seg}_labeled.nii.gz -l ${file_compression}.nii.gz -normalize 1 -metric solidity -o ${PATH_RESULTS}/${file_t2_ax}_solidity_norm.csv
    sct_compute_compression -i ${file_t2_ax_seg}.nii.gz -vertfile ${file_t2_ax_seg}_labeled.nii.gz -l ${file_compression}.nii.gz -normalize 0 -metric solidity -o ${PATH_RESULTS}/${file_t2_ax}_solidity.csv
fi

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
