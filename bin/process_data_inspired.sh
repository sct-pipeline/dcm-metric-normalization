#!/bin/bash
#
# Run `sct_process_segmentation -normalize PAM50` on axial T2w images from HC and SCI and DCM patients (INSPIRED dataset)
# SC segmentations from /derivatives are used
# Note: files in /derivatives were created from "raw" axial T2w images (i.e., without any preprocessing steps). Thus, no
# preprocessing steps are used also within this script.

# Usage:
#     sct_run_batch -c <PATH_TO_REPO>/etc/config_process_data.json

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
  FILESEGMANUAL="${PATH_DATA}/derivatives/manual_labels/${SUBJECT}/anat/${FILESEG}.nii.gz"
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
  FILELABELMANUAL="${PATH_DATA}/derivatives/manual_labels/${SUBJECT}/anat/${FILELABEL}.nii.gz"
  echo "Looking for manual label: $FILELABELMANUAL"
  if [[ -e $FILELABELMANUAL ]]; then
    echo "Found! Using manual labels."
    rsync -avzh $FILELABELMANUAL ${FILELABEL}.nii.gz
    # Generate labeled segmentation from manual disc labels
    sct_label_vertebrae -i ${file}.nii.gz -s ${file_seg}.nii.gz -discfile ${FILELABEL}.nii.gz -c ${contrast} -qc ${PATH_QC} -qc-subject ${SUBJECT}
  else
    echo "Not found. Proceeding with automatic labeling."
    # Generate labeled segmentation automatically (no manual disc labels provided)
    sct_label_vertebrae -i ${file}.nii.gz -s ${file_seg}.nii.gz -c ${contrast} -qc ${PATH_QC} -qc-subject ${SUBJECT}
  fi
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
file_t2_ax="${SUBJECT//[\/]/_}"_acq-cspineAxial_T2w

# Note: manual segmentations and disc labels located under /derivatives were created from "raw" images without any
# preprocessing. Thus, no preprocessing steps are applied also here.

# Copy SC segmentation from /derivatives
segment_if_does_not_exist ${file_t2_ax} 't2'
file_t2_ax_seg=$FILESEG
# Create native labeling from manual disc labels located under /derivatives
# Note: output of the following command does not include levels above the top label and below the bottom label
label_if_does_not_exist ${file_t2_ax} ${file_t2_ax_seg} 't2'

# Thus, method using PAM50 template is tried
sct_register_to_template -i ${file_t2_ax}.nii.gz -s ${file_t2_ax_seg}.nii.gz -ldisc ${file_t2_ax}_label-disc.nii.gz -ref template -c t2 -param step=1,type=seg,algo=centermassrot:step=2,type=seg,algo=syn,slicewise=1,smooth=0,iter=5:step=3,type=im,algo=syn,slicewise=1,smooth=0,iter=3
# Rename warping fields for clarity
mv warp_template2anat.nii.gz warp_template2Axial_T2w.nii.gz
mv warp_anat2template.nii.gz warp_Axial_T2w2template.nii.gz
# Warp PAM50 vertebral labeling (-a 0: we don't need WM atlas)
sct_warp_template -d ${file_t2_ax}.nii.gz -w warp_template2Axial_T2w.nii.gz -a 0 -ofolder label_Axial_T2w
# Generate QC report to assess vertebral labeling
sct_qc -i ${file_t2_ax}.nii.gz -s label_Axial_T2w/template/PAM50_levels.nii.gz -p sct_label_vertebrae -qc ${PATH_QC} -qc-subject ${SUBJECT}

# Bring vertebral labeling from PAM50
sct_register_multimodal -i label_Sagittal_T2w/template/PAM50_levels.nii.gz -d ${file_t2_ax}.nii.gz -o PAM50_levels2Axial_T2w.nii.gz -identity 1 -x nn
# Generate QC report to assess vertebral labeling
sct_qc -i ${file_t2_ax}.nii.gz -s PAM50_levels2Axial_T2w.nii.gz -p sct_label_vertebrae -qc ${PATH_QC} -qc-subject ${SUBJECT}

# Compute metrics from SC segmentation and normalize them to PAM50 (`-normalize PAM50` flag)
sct_process_segmentation -i ${file_t2_ax_seg}.nii.gz -perslice 1 -vert 1:20 -vertfile ${file_t2_ax_seg}_labeled.nii.gz -o ${PATH_RESULTS}/${file_t2_ax}_native_labeling.csv -normalize PAM50
sct_process_segmentation -i ${file_t2_ax_seg}.nii.gz -perslice 1 -vert 1:20 -vertfile label_Axial_T2w/template/PAM50_levels.nii.gz -o ${PATH_RESULTS}/${file_t2_ax}_PAM50_labeling.csv -normalize PAM50

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
