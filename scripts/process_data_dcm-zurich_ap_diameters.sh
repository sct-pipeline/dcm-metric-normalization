#!/bin/bash
#
# Process dcm-zurich dataset: compute anterior and posterior cord lengths from T2w axial images.
#
# For each subject/session (using ses-M0 here), the script:
#   1. Segments the spinal cord (uses manual segmentation from derivatives/labels if available,
#      otherwise runs sct_deepseg spinalcord automatically)
#   2. Labels intervertebral discs (uses manual labels from derivatives/labels if available,
#      otherwise runs TotalSpineSeg automatically)
#   3. Computes per-slice cord morphometrics (including anterior and posterior lengths)
#      and appends results to T2w_ax_cord_metrics_perlevel.csv
#
# Usage to exclude specific subjects:
#     sct_run_batch -c config_process_data_dcm-zurich_ap_diameters.json -exclude-yml exclude_dcm-zurich.yml
#
# Example YAML exclude file (included in this repo) to exclude some subjects/sessions (note that YAML supports comments):
#     t2_ax:
#       - sub-042/ses-M0    # missing T2w ax image
#       - sub-045/ses-M0    # missing T2w ax image
#
# Example JSON configuration file to process only session M0:
#   {
#     "path_data"   : "~/data/dcm-zurich",
#     "path_output" : "~/results/dcm-zurich/dcm-zurich_2026-04-09",
#     "script"      : "~/code/dcm-metric-normalization/scripts/process_data_dcm-zurich_ap_diameters.sh",
#     "jobs"        : 8,
#     "include"     : "ses-M0"
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
# If it doesn't, perform automatic spinal cord segmentation
segment_if_does_not_exist() {
  local file="$1"
  local contrast="$2"   # only for logging
  # Update global variable with segmentation file name
  FILESEG="${file}_label-SC_seg"
  # Getting the path for manual segmentation for each session
  FILESEGMANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/${SESSION}/anat/${FILESEG}.nii.gz"
  echo
  echo "Looking for manual segmentation: $FILESEGMANUAL"
  if [[ -e $FILESEGMANUAL ]]; then
    echo "✅ [$(date '+%Y-%m-%d %H:%M:%S')] Found! Using manual spinal cord segmentation."
    echo "✅ [$(date '+%Y-%m-%d %H:%M:%S')] ${FILESEG}.nii.gz found under derivatives/labels --> using manual spinal cord segmentation" >> "${PATH_LOG}/${contrast}_SC_segmentations.log"
    rsync -avzh $FILESEGMANUAL ${FILESEG}.nii.gz
    sct_qc -i ${file}.nii.gz -s ${FILESEG}.nii.gz -p sct_deepseg_sc -qc ${PATH_QC} -qc-subject ${SUBJECT}_${SESSION}
  else
    echo "🤖 [$(date '+%Y-%m-%d %H:%M:%S')] Not found. Proceeding with automatic spinal cord segmentation."
    echo "🤖 [$(date '+%Y-%m-%d %H:%M:%S')] ${FILESEG}.nii.gz NOT found --> segmenting spinal cord automatically" >> "${PATH_LOG}/${contrast}_SC_segmentations.log"
    # Segment spinal cord
    sct_deepseg spinalcord -i ${file}.nii.gz -o ${FILESEG}.nii.gz -qc ${PATH_QC} -qc-subject ${SUBJECT}_${SESSION}
  fi
}

label_t2_ax_if_does_not_exist(){
  local file="$1"
  # Copy manual disc labels from derivatives/labels if they exist
  FILELABEL="${file}_label-disc"
  FILELABELMANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/${SESSION}/anat/${file}_labels-manual.nii.gz"
  echo "Looking for manual disc labels: $FILELABELMANUAL"
  if [[ -e $FILELABELMANUAL ]]; then
    echo "✅ [$(date '+%Y-%m-%d %H:%M:%S')] Found! Using manual disc labels."
    echo "✅ [$(date '+%Y-%m-%d %H:%M:%S')] ${FILELABEL}.nii.gz found under derivatives/labels --> using manual disc labels" >> "${PATH_LOG}/T2w_ax_disc_labels.log"
    rsync -avzh $FILELABELMANUAL ${FILELABEL}.nii.gz
  else
    echo "🤖 [$(date '+%Y-%m-%d %H:%M:%S')] Manual disc labels not found. Proceeding with automatic labeling using TotalSpineSeg."
    echo "🤖 [$(date '+%Y-%m-%d %H:%M:%S')] ${FILELABEL}.nii.gz NOT found --> using automatic labeling using TotalSpineSeg" >> "${PATH_LOG}/T2w_ax_disc_labels.log"
    # Automatically label discs using TotalSpineSeg
    sct_deepseg totalspineseg -i ${file}.nii.gz -step1-only 1
    # Keep only disc labels and remove other outputs
    mv ${file}_step1_levels.nii.gz ${FILELABEL}.nii.gz
    rm ${file}_step1_cord.nii.gz ${file}_step1_canal.nii.gz ${file}_step1_output.nii.gz
    # Create disc labels QC
    sct_qc -i ${file}.nii.gz -s ${FILELABEL}.nii.gz -p sct_label_utils -qc ${PATH_QC} -qc-subject "disc_labels_totalspineseg"
  fi
}

# Retrieve input params and other params
SUBJECT_INPUT=$1
# Adapting the script to handle new folder organization where each subject has multiple sessions
SESSION_INPUT=${2:-ses-M0}  # Default to ses-M0 if not provided

# Handle sct_run_batch format where SUBJECT might be "sub-001/ses-M0"
if [[ "$SUBJECT_INPUT" == *"/"* ]]; then
    # Extract subject and session from combined format (sct_run_batch style)
    SUBJECT=$(echo "$SUBJECT_INPUT" | cut -d'/' -f1)
    SESSION=$(echo "$SUBJECT_INPUT" | cut -d'/' -f2)
    echo "Detected sct_run_batch format: $SUBJECT_INPUT -> SUBJECT=$SUBJECT, SESSION=$SESSION"
else
    # Use separate parameters (direct script call)
    SUBJECT=$SUBJECT_INPUT
    SESSION=$SESSION_INPUT
fi

# Verify the session directory exists (only if PATH_DATA is set)
if [[ -n "${PATH_DATA}" && ! -d "${PATH_DATA}/${SUBJECT}/${SESSION}" ]]; then
    echo "ERROR: Session directory ${PATH_DATA}/${SUBJECT}/${SESSION} does not exist"
    echo "Available sessions for ${SUBJECT}:"
    find ${PATH_DATA}/${SUBJECT} -maxdepth 1 -type d -name "ses-*" 2>/dev/null | sort || echo "  No sessions found or subject doesn't exist"
    exit 1
fi

echo "Processing ${SUBJECT} - ${SESSION}"

# get starting time:
start=`date +%s`

# ------------------------------------------------------------------------------
# SCRIPT STARTS HERE
# ------------------------------------------------------------------------------
# Display useful info for the log, such as SCT version, RAM and CPU cores available
sct_check_dependencies -short

# Go to folder where data will be copied and processed
cd $PATH_DATA_PROCESSED

# Copy participants.tsv file (will be used to fetch sex) to the root folder of the dataset
PARTICIPANTS_PATH=${PATH_DATA_PROCESSED}/../participants.tsv
if [[ ! -e ../participants.tsv ]]; then
    rsync -avzh ${PATH_DATA}/participants.tsv ${PARTICIPANTS_PATH}
fi

# Copy source T2w images
# Note: we use '/./' in order to include the sub-folder 'ses-0X'
rsync -Ravzh ${PATH_DATA}/./${SUBJECT}/${SESSION}/anat/${SUBJECT}_${SESSION}*T2w.* .

# Go to subject folder for source images
cd ${SUBJECT}/${SESSION}/anat

# ------------------------------------------------------------------------------
# T2w Axial
# ------------------------------------------------------------------------------
# Define variables
# We do a substitution '/' --> '_' in case there is a subfolder 'ses-0X/'
file_t2_ax="${SUBJECT//[\/]/_}"_"${SESSION}"_acq-axial_T2w
# Check if file_t2_ax exists.
# Note: some subjects do not have T2w axial images. In this case, analysis will be stop after processing of
# T2w sagittal image.
if [[ ! -e ${file_t2_ax}.nii.gz ]]; then
    echo "File ${file_t2_ax}.nii.gz does not exist" >> ${PATH_LOG}/missing_files.log
    echo "ERROR: File ${file_t2_ax}.nii.gz does not exist. Exiting."
    exit 1
else
    # -------------
    # Segment SC (if SC segmentation file already exists under derivatives folder, it will be copied)
    # -------------
    segment_if_does_not_exist ${file_t2_ax} 'T2w_ax'
    file_t2_ax_seg=$FILESEG

    label_t2_ax_if_does_not_exist ${file_t2_ax}
    file_t2_ax_labels=${file_t2_ax}_label-disc

    # -------------
    # Compute anterior and posterior lengths
    # https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/5199
    # -------------
    echo "Computing spinal cord morphometrics..."
    # Compute cord metrics perslice in the native space -- metrics across subjects are appended to a single CSV file
    # Note: -anat is used for QC purposes to display the metrics on the original image space.
    sct_process_segmentation -anat ${file_t2_ax}.nii.gz -i ${file_t2_ax_seg}.nii.gz -discfile ${file_t2_ax_labels}.nii.gz -perslice 1 -vert 2:9 -o ${PATH_RESULTS}/T2w_ax_cord_metrics_perlevel.csv -append 1 -qc ${PATH_QC}

    echo "Finished processing ${file_t2_ax}" >> ${PATH_LOG}/processed_files_T2w_ax.log

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
