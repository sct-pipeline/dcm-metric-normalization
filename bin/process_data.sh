#!/bin/bash
#
# Preprocess data.

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
    sct_deepseg_sc -i ${file}.nii.gz -c ${contrast} -qc ${PATH_QC} -qc-subject ${SUBJECT}
  fi
}

# Check if manual label already exists. If it does, copy it locally. If it does
# not, perform labeling.
# NOTE: manual disc labels should go from C1-C2 to C7-T1.
label_if_does_not_exist(){
  local file="$1"
  local file_seg="$2"
  local contrast="$3"
  # Update global variable with segmentation file name
  FILELABEL="${file}_labels-disc"
  FILELABELMANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/${FILELABEL}-manual.nii.gz"
  # Binarize softsegmentation to create labeled softseg
  #sct_maths -i ${file_seg}.nii.gz -bin 0.5 -o ${file_seg}_bin.nii.gz
  echo "Looking for manual label: $FILELABELMANUAL"
  if [[ -e $FILELABELMANUAL ]]; then
    echo "Found! Using manual labels."
    rsync -avzh $FILELABELMANUAL ${FILELABEL}.nii.gz
    # Generate labeled segmentation from manual disc labels
    sct_label_vertebrae -i ${file}.nii.gz -s ${file_seg}.nii.gz -discfile ${FILELABEL}.nii.gz -c t2 -ofolder ./anat/
  else
    echo "Not found. Proceeding with automatic labeling."
    # Generate labeled segmentation
    sct_label_vertebrae -i ${file}.nii.gz -s ${file_seg}.nii.gz -c ${contrast} -ofolder ./anat/
  fi
}

# Retrieve input params and other params
SUBJECT=$1

# get starting time:
start=`date +%s`


# SCRIPT STARTS HERE
# ==============================================================================
# Display useful info for the log, such as SCT version, RAM and CPU cores available
sct_check_dependencies -short

# Go to folder where data will be copied and processed
cd $PATH_DATA_PROCESSED

# Copy source images
# Note: we use '/./' in order to include the sub-folder 'ses-0X'
rsync -Ravzh $PATH_DATA/./$SUBJECT .

# Go to subject folder for source images
cd ${SUBJECT}/anat

# ------------------------------------------------------------------------------
# T2w
# ------------------------------------------------------------------------------
# Define variables
# We do a substitution '/' --> '_' in case there is a subfolder 'ses-0X/'
file_t2="${SUBJECT//[\/]/_}"

# Reorient to RPI and resample to 0.8mm iso (supposed to be the effective resolution)
sct_image -i ${file_t2}.nii.gz -setorient RPI -o ${file_t2}_RPI.nii.gz
sct_resample -i ${file_t2}_RPI.nii.gz -mm 0.8x0.8x0.8 -o ${file_t2}_RPI_r.nii.gz
file_t2="${file_t2}_RPI_r"

segment_if_does_not_exist ${file_t2} 't2'
file_t2_seg="${file_t2}_seg"
label_if_does_not_exist ${file_t2} ${file_t2_seg} 't2'

# Compute metrics from SC segmentation in native space
sct_process_segmentation -i ${file_t2_seg}.nii.gz -perslice 1 -vertfile ${file_t2_seg}_labeled.nii.gz -o ${PATH_RESULTS}/${file_t2}_native.csv

# Register t2 image to PAM50 template
# NOTES:
#   `-ldisc`            --> we are using more than 2 labels (i.e., labels for all discs)
#   `-ref subject`      --> no SC straightening --> better for axial images with anisotropic resolution to avoid interpolation errors
#   `dof=Tx_Ty_Tz_Sz`   --> allow scaling only in S-I (z-axis) direction (to do not change the shape of compressed spinal cord); we do not want to do rotation (R) since we want to compute torsion (which si computed from orientation between adjacent slices)
#   `algo=centermass`   --> again, we do not want to do rotation, thus we use slicereg algorithm, which uses just translation in x,y-axes
# TODO - although `algo=centermass` should NOT do a rotation (because the rotation is included in centermassrot), the dof printed to the CLI during step=1 are: Tx_Ty_Tz_Rx_Ry_Rz. Thus we tried to specify dof manually. --> compare if there is any difference
sct_register_to_template -i ${file_t2}.nii.gz -s ${file_t2_seg}.nii.gz -ldisc labels.nii.gz -ref subject -c t2 -param step=0,type=label,dof=Tx_Ty_Tz_Sz:step=1,type=seg,algo=centermass -ofolder ref_subject_centermass -qc ${PATH_QC} -qc-subject ${SUBJECT}
sct_register_to_template -i ${file_t2}.nii.gz -s ${file_t2_seg}.nii.gz -ldisc labels.nii.gz -ref subject -c t2 -param step=0,type=label,dof=Tx_Ty_Tz_Sz:step=1,type=seg,algo=centermass,dof=Tx_Ty_Tz -ofolder ref_subject_centermass_dof_Tx_Ty_Tz -qc ${PATH_QC} -qc-subject ${SUBJECT}

# Bring SC segmentation to PAM50
sct_apply_transfo -i ${file_t2_seg}.nii.gz -d $SCT_DIR/data/PAM50/template/PAM50_t2.nii.gz -w warp_anat2template.nii.gz

# Compute metrics from SC segmentation in PAM50 space
sct_process_segmentation -i ${file_t2_seg}_reg.nii.gz -perslice 1 -vertfile $SCT_DIR/data/PAM50/template/PAM50_levels.nii.gz -o ${PATH_RESULTS}/${file_t2}_pam50.csv

