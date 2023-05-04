#!/bin/bash
#
# Process dcm-brno dataset and run `sct_compute_compression` on T2star axial images
#
# Usage:
#     sct_run_batch -c <PATH_TO_REPO>/etc/config_process_data_<DATASET>.json
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

# Define path to manual labels
PATH_SEGMANUAL="${PATH_DATA}/derivatives/labels/"

echo "Retrieved variables from from the caller sct_run_batch:"
echo "PATH_DATA: ${PATH_DATA}"
echo "PATH_DATA_PROCESSED: ${PATH_DATA_PROCESSED}"
echo "PATH_RESULTS: ${PATH_RESULTS}"
echo "PATH_LOG: ${PATH_LOG}"
echo "PATH_QC: ${PATH_QC}"

# ------------------------------------------------------------------------------
#  CONVENIENCE FUNCTIONS
# ------------------------------------------------------------------------------
# Check if manual spinal cord segmentation file already exists. If it does, copy it locally.
# If it doesn't, perform automatic spinal cord segmentation
segment_if_does_not_exist() {
  local file="$1"
  local contrast="$2"
  # Update global variable with segmentation file name
  FILESEG="${file}_seg"
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
  FILELABEL="${file/_RPI_r/}_labels-disc"
  FILELABELMANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${FILELABEL}-manual.nii.gz"
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
rsync -Ravzh ${PATH_DATA}/./${SUBJECT}/anat/* .

# Go to subject folder for source images
cd ${SUBJECT}/anat

# ------------------------------------------------------------------------------
# T2w Sagittal
# ------------------------------------------------------------------------------
# Define variables
# We do a substitution '/' --> '_' in case there is a subfolder 'ses-0X/'
file_t2="${SUBJECT//[\/]/_}"_T2w
# Check if file_t2_sag exists
if [[ ! -e ${file_t2}.nii.gz ]]; then
    echo "File ${file_t2}.nii.gz does not exist" >> ${PATH_LOG}/missing_files.log
    echo "ERROR: File ${file_t2}.nii.gz does not exist. Exiting."
    exit 1
else
    # Segment SC
    segment_if_does_not_exist ${file_t2} 't2'
    file_t2_seg=$FILESEG
    label_if_does_not_exist ${file_t2} ${file_t2_seg} 't2'
fi

# ------------------------------------------------------------------------------
# T2star Axial
# ------------------------------------------------------------------------------
# Define variables
# We do a substitution '/' --> '_' in case there is a subfolder 'ses-0X/'
file_t2star="${SUBJECT//[\/]/_}"_acq-2-5mm_T2star

# Note: some subjects do not have T2star images. In this case, analysis will be stop after processing of
# T2w sagittal image.
if [[ ! -e ${file_t2star}.nii.gz ]]; then
    echo "File ${file_t2star}.nii.gz does not exist" >> ${PATH_LOG}/missing_files.log
    echo "ERROR: File ${file_t2star}.nii.gz does not exist. Exiting."
    exit 1
else
    # Segment SC (if SC segmentation file already exists under derivatives folder, it will be copied)
    segment_if_does_not_exist ${file_t2star} 't2'
    file_t2star_seg=$FILESEG

    # Label SC
    # Check if manual disc labels file already exists. If so, generate labeled segmentation from manual disc labels.
    echo "Looking for manual disc labels: ${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${file_t2star}_labels-manual.nii.gz"
    if [[ -e ${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${file_t2star}_labels-manual.nii.gz ]]; then
        echo "Found! Using manual disc labels."
        rsync -avzh ${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${file_t2star}_labels-manual.nii.gz ${file_t2star}_labels.nii.gz

        file_t2star_labels=${file_t2star}_labels
    # If manual disc labels file does not exist, use disc labels from sagittal image
    else
        # Bring T2w sagittal image to T2star axial image to obtain warping field.
        # This warping field will be used to bring the T2w sagittal disc labels to the T2star axial space.
        # Context: https://github.com/sct-pipeline/dcm-metric-normalization/issues/9
        # Note: the '-dseg' is used only for the QC report
        sct_register_multimodal -i ${file_t2}.nii.gz -d ${file_t2star}.nii.gz -identity 1 -x nn -qc ${PATH_QC} -qc-subject ${SUBJECT} -dseg ${file_t2star_seg}.nii.gz
        # Bring T2w sagittal disc labels (located in the middle of the spinal cord) to T2star axial space
        # Context: https://github.com/sct-pipeline/dcm-metric-normalization/issues/10
        sct_apply_transfo -i ${file_t2_seg}_labeled_discs.nii.gz -d ${file_t2star}.nii.gz -w warp_${file_t2}2${file_t2star}.nii.gz -x label
        # Generate QC report to assess warped disc labels
        sct_qc -i ${file_t2star}.nii.gz -s ${file_t2_seg}_labeled_discs_reg.nii.gz -p sct_label_utils -qc ${PATH_QC} -qc-subject ${SUBJECT}

        file_t2star_labels=${file_t2_seg}_labeled_discs_reg
    fi

    # Label T2star axial spinal cord segmentation. Either using manual disc labels or using disc labels from sagittal image.
    # Note: here we use sct_label_utils instead of sct_label_vertebrae to avoid SC straightening
    # Context: https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4072
    sct_label_utils -i ${file_t2star_seg}.nii.gz -disc ${file_t2star_labels}.nii.gz -o ${file_t2star_seg}_labeled.nii.gz
    # Generate QC report to assess labeled segmentation
    sct_qc -i ${file_t2star}.nii.gz -s ${file_t2star_seg}_labeled.nii.gz -p sct_label_vertebrae -qc ${PATH_QC} -qc-subject ${SUBJECT}

    # TODO: add sct_compute_compression
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