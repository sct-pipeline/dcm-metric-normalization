#!/bin/bash
#
# Run `sct_process_segmentation -normalize PAM50` on T2w images from DCM patients (dcm-zurich dataset)
# The script also computes normalized MSCC using `sct_compute_mscc`
#
# Note: Since axial T2w images have limited FOV (C2-C7) and high slice thickness (3.6mm), the vertebral labeling from
# sagittal T2w images is brought to axial T2w images
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
  # TODO - change to FILESEG="${file}_label-SC_seg", once https://github.com/neuropoly/data-management/issues/225 is done
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
  FILELABEL="${file}_labels"
  # TODO - change to FILELABEL="${file}_label-disc", once https://github.com/neuropoly/data-management/issues/225 is done
  FILELABELMANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${FILELABEL}-manual.nii.gz"
  echo "Looking for manual label: $FILELABELMANUAL"
  if [[ -e $FILELABELMANUAL ]]; then
    echo "Found! Using manual labels."
    rsync -avzh $FILELABELMANUAL ${FILELABEL}.nii.gz
    # Generate labeled segmentation from manual disc labels
    sct_label_vertebrae -i ${file}.nii.gz -s ${file_seg}.nii.gz -discfile ${FILELABEL}.nii.gz -c ${contrast} -qc ${PATH_QC} -qc-subject ${SUBJECT}
    # Generate QC to access manual disc labels
    sct_qc -i ${file}.nii.gz -s ${FILELABEL}.nii.gz -p sct_label_utils -qc ${PATH_QC} -qc-subject ${SUBJECT}
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
# T2w Sagittal
# ------------------------------------------------------------------------------
# Define variables
# We do a substitution '/' --> '_' in case there is a subfolder 'ses-0X/'
file_t2_sag="${SUBJECT//[\/]/_}"_acq-sagittal_T2w
# Check if file_t2_sag exists
if [[ ! -e ${file_t2_sag}.nii.gz ]]; then
    echo "File ${file_t2_sag}.nii.gz does not exist" >> ${PATH_LOG}/missing_files.log
    echo "ERROR: File ${file_t2_sag}.nii.gz does not exist. Exiting."
    exit 1
else
    # Segment SC
    segment_if_does_not_exist ${file_t2_sag} 't2'
    file_t2_sag_seg=$FILESEG
    label_if_does_not_exist ${file_t2_sag} ${file_t2_sag_seg} 't2'
fi
# ------------------------------------------------------------------------------
# T2w Axial
# ------------------------------------------------------------------------------
# Define variables
# We do a substitution '/' --> '_' in case there is a subfolder 'ses-0X/'
file_t2_ax="${SUBJECT//[\/]/_}"_acq-axial_T2w
# Check if file_t2_ax exists.
# Note: some subjects do not have T2w axial images. In this case, analysis will be stop after processing of
# T2w sagittal image.
if [[ ! -e ${file_t2_ax}.nii.gz ]]; then
    echo "File ${file_t2_ax}.nii.gz does not exist" >> ${PATH_LOG}/missing_files.log
    echo "ERROR: File ${file_t2_ax}.nii.gz does not exist. Exiting."
    exit 1
else
    # Segment SC (if SC segmentation file already exists under derivatives folder, it will be copied)
    segment_if_does_not_exist ${file_t2_ax} 't2'
    file_t2_ax_seg=$FILESEG

    # Bring T2w sagittal image to T2w axial image to obtain warping field.
    # This warping field will be used to bring the T2w sagittal disc labels to the T2w axial space.
    sct_register_multimodal -i ${file_t2_sag}.nii.gz -d ${file_t2_ax}.nii.gz -identity 1 -x nn
    # TODO - change to "${file_t2_sag}_label-disc.nii.gz", once https://github.com/neuropoly/data-management/issues/225 is done
    sct_apply_transfo -i ${file_t2_sag}_labels.nii.gz -d ${file_t2_ax}.nii.gz -w warp_{file_t2_sag}2${file_t2_ax}.nii.gz -x label
    # Generate QC report to assess warped disc labels
    sct_qc -i ${file_t2_ax}.nii.gz -s ${file_t2_sag}_labels_reg.nii.gz -p sct_label_utils -qc ${PATH_QC} -qc-subject ${SUBJECT}

    # Now, label T2w axial spinal cord segmentation using warped T2w sagittal disc labels
    sct_label_utils.py -i ${file_t2_ax_seg}.nii.gz -disc ${file_t2_sag}_labels_reg.nii.gz -o ${file_t2_ax_seg}_labeled.nii.gz
    # Generate QC report to assess labeled segmentation
    sct_qc -i ${file_t2_ax}.nii.gz -s ${file_t2_ax_seg}_labeled.nii.gz -p sct_label_vertebrae -qc ${PATH_QC} -qc-subject ${SUBJECT}

    # Compute metrics from SC segmentation and normalize them to PAM50 ('-normalize-PAM50 1')
    sct_process_segmentation -i ${file_t2_ax_seg}.nii.gz -vertfile ${file_t2_ax_seg}_labeled.nii.gz -perslice 1 -normalize-PAM50 1 -o ${PATH_RESULTS}/${file_t2_ax}_PAM50_space.csv
    # Compute metrics from SC segmentation without PAM50 normalization ('-normalize-PAM50 0')
    sct_process_segmentation -i ${file_t2_ax_seg}.nii.gz -vertfile ${file_t2_ax_seg}_labeled.nii.gz -perslice 1 -normalize-PAM50 0 -o ${PATH_RESULTS}/${file_t2_ax}_subject_space.csv

    # TODO, add sct_compute_compression once https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4003 is merged
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
