#!/bin/bash
#
# Process dcm-zurich dataset
#
# Requirements: SCT v7.3 (with sct_process_segmentation supporting symmetry measures and updated AP diameter)
#
# Usage to exclude specific subjects:
#     sct_run_batch -c config_process_data_dcm-zurich.json -exclude-yml ~/data/dcm-zurich/exclude.yml
#
# Example config_process_data_dcm-zurich.json configuration file to process only session M0:
#   {
#     "path_data"   : "~/data/dcm-zurich",
#     "path_output" : "~/results/dcm-zurich/dcm-zurich_2026-05-14",
#     "script"      : "~/code/dcm-metric-normalization/scripts/process_data_dcm-zurich.sh",
#     "jobs"        : 8,
#     "include"     : "ses-M0"
#   }
#
# NOTE that both "include" and "exclude-yml" options can be used together.
#
# The following global variables are retrieved from the caller sct_run_batch
# but could be overwritten by uncommenting the lines below:
# PATH_DATA_PROCESSED="~/data_processed"
# PATH_RESULTS="~/results"
# PATH_LOG="~/log"
# PATH_QC="~/qc"
#
# Authors: Jan Valosek, Sandrine Bedard, Kahina, Julien Cohen-Adad
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

# Check if manual T2w sag disc labels already exist. If it does, generate labeled segmentation from manual disc labels.
# If it doesn't, perform automatic spinal cord labeling
label_t2_sag_if_does_not_exist(){
  local file="$1"
  local file_seg="$2"
  local contrast="$3"
  # Update global variable with segmentation file name
  FILELABEL="${file}_labels"
  FILELABELMANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/${SESSION}/anat/${FILELABEL}-manual.nii.gz"
  echo "Looking for manual disc labels: $FILELABELMANUAL"
  if [[ -e $FILELABELMANUAL ]]; then
    echo "✅ [$(date '+%Y-%m-%d %H:%M:%S')] Found! Using manual T2w sag disc labels."
    echo "✅ [$(date '+%Y-%m-%d %H:%M:%S')] ${FILELABEL}.nii.gz found --> using manual T2w sag disc labels" >> "${PATH_LOG}/T2w_sag_disc_labels.log"
    rsync -avzh $FILELABELMANUAL ${FILELABEL}.nii.gz
    # Generate labeled segmentation from manual disc labels
    sct_label_vertebrae -i ${file}.nii.gz -s ${file_seg}.nii.gz -discfile ${FILELABEL}.nii.gz -c ${contrast} -qc ${PATH_QC} -qc-subject ${SUBJECT}_${SESSION}
  else
    echo "🤖 [$(date '+%Y-%m-%d %H:%M:%S')] Manual T2w sag disc labels not found. Proceeding with automatic labeling."
    echo "🤖 [$(date '+%Y-%m-%d %H:%M:%S')] ${FILELABEL}.nii.gz NOT found --> using automatic T2w sag labeling" >> "${PATH_LOG}/T2w_sag_disc_labels.log"
    # Generate labeled segmentation automatically (no manual disc labels provided)
    sct_label_vertebrae -i ${file}.nii.gz -s ${file_seg}.nii.gz -c ${contrast} -qc ${PATH_QC} -qc-subject ${SUBJECT}_${SESSION}
  fi
  # Generate QC to access disc labels created by sct_label_vertebrae
  sct_qc -i ${file}.nii.gz -s ${file_seg}_labeled_discs.nii.gz -p sct_label_utils -qc ${PATH_QC} -qc-subject ${SUBJECT}_${SESSION}
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

# Check if manual canal segmentation file already exists. If it does, copy it locally.
# If it doesn't, perform automatic canal segmentation
segment_canal_if_does_not_exist() {
  local file="$1"
  # Update global variable with segmentation file name 
  FILESEG="${file}_label-canal_seg"
  FILESEGMANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/${SESSION}/anat/${FILESEG}.nii.gz"
  echo "Looking for manual canal segmentation: $FILESEGMANUAL"
  if [[ -e $FILESEGMANUAL ]]; then
    echo "✅ [$(date '+%Y-%m-%d %H:%M:%S')] Found! Using manual canal segmentation."
    echo "✅ [$(date '+%Y-%m-%d %H:%M:%S')] ${FILESEG}.nii.gz found under derivatives/labels --> using manual canal segmentation" >> "${PATH_LOG}/T2w_canal_segmentations.log"
    rsync -avzh $FILESEGMANUAL ${FILESEG}.nii.gz
    sct_qc -i ${file}.nii.gz -s ${FILESEG}.nii.gz -p sct_deepseg_sc -qc ${PATH_QC} -qc-subject ${SUBJECT}_${SESSION}
  else
    echo "🤖 [$(date '+%Y-%m-%d %H:%M:%S')] Not found. Proceeding with automatic canal segmentation."
    echo "🤖 [$(date '+%Y-%m-%d %H:%M:%S')] ${FILESEG}.nii.gz NOT found --> segmenting canal automatically" >> "${PATH_LOG}/T2w_canal_segmentations.log"
    # Segment canal
    sct_deepseg sc_canal_t2 -i ${file}.nii.gz -o ${FILESEG}.nii.gz -qc ${PATH_QC} -qc-subject ${SUBJECT}_${SESSION} -largest 1
  fi
}

# Check if manual lesion segmentation file already exists. If it does, copy it locally.
# If it doesn't, perform automatic lesion segmentation
segment_lesion_if_does_not_exist() {
  local file="$1"
  local contrast="$2"
  # Update global variable with segmentation file name 
  FILESEG="${file}"
  FILESEGMANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/${SESSION}/anat/${file}_label-lesion_seg.nii.gz"
  echo "Looking for manual lesion segmentation: $FILESEGMANUAL"
  if [[ -e $FILESEGMANUAL ]]; then
    echo "✅ [$(date '+%Y-%m-%d %H:%M:%S')] Found! Using manual lesion segmentation."
    echo "✅ [$(date '+%Y-%m-%d %H:%M:%S')] ${file}_lesion_seg.nii.gz found under derivatives/labels --> using manual lesion segmentation" >> "${PATH_LOG}/T2w_lesion_segmentations.log"
    rsync -avzh $FILESEGMANUAL ${file}_lesion_seg.nii.gz
    sct_qc -i ${file}.nii.gz -s ${file}_lesion_seg.nii.gz -p sct_deepseg_lesion -qc ${PATH_QC} -qc-subject ${SUBJECT}_${SESSION}
  else
    echo "🤖 [$(date '+%Y-%m-%d %H:%M:%S')] Not found. Proceeding with automatic lesion segmentation."
    echo "🤖 [$(date '+%Y-%m-%d %H:%M:%S')] ${file}_lesion_seg.nii.gz NOT found --> segmenting lesion automatically" >> "${PATH_LOG}/T2w_lesion_segmentations.log"
    # Segment lesions
    sct_deepseg lesion_sci_t2 -i ${file}.nii.gz -o ${file}.nii.gz -c ${contrast} -qc ${PATH_QC} -qc-subject ${SUBJECT}_${SESSION}
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

## ------------------------------------------------------------------------------
## T2w Sagittal
## ------------------------------------------------------------------------------
## Define variables
## We do a substitution '/' --> '_' in case there is a subfolder 'ses-0X/'
#file_t2_sag="${SUBJECT//[\/]/_}"_"${SESSION}"_acq-sagittal_T2w
## Check if file_t2_sag exists
#if [[ ! -e ${file_t2_sag}.nii.gz ]]; then
#    echo "File ${file_t2_sag}.nii.gz does not exist" >> ${PATH_LOG}/missing_files.log
#    echo "ERROR: File ${file_t2_sag}.nii.gz does not exist. Exiting."
#    exit 1
#else
#    # Segment SC
#    segment_if_does_not_exist ${file_t2_sag} 'T2w_sag'
#    file_t2_sag_seg=$FILESEG
#    label_t2_sag_if_does_not_exist ${file_t2_sag} ${file_t2_sag_seg} 't2'
#
#    echo "Finished processing ${file_t2_sag}" >> ${PATH_LOG}/processed_files_T2w_sag.log
#
#fi
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
    # Compute spinal cord morphometrics
    # -------------
    # Note: '-anat' flag is used to specify the anatomical image to compute spinal cord orientation (using HOG method).
    #  It is required to compute symmetry and quadrants area metrics.
    echo "Computing spinal cord morphometrics..."
    # Compute cord metrics perlevel in the native space -- metrics across subjects are appended to a single CSV file
    sct_process_segmentation -i ${file_t2_ax_seg}.nii.gz -discfile ${file_t2_ax_labels}.nii.gz -anat {file_t2_ax}.nii.gz -perlevel 1 -vert 2:9 -o ${PATH_RESULTS}/T2w_ax_cord_metrics_perlevel.csv -append 1
    sct_qc -i ${file_t2_ax}.nii.gz -s ${file_t2_ax_labels}_projected_centerline.nii.gz -p sct_label_vertebrae -qc ${PATH_QC} -qc-subject ${SUBJECT}_${SESSION}
    # Compute cord metrics perslice in the native space -- metrics across subjects are appended to a single CSV file
    sct_process_segmentation -i ${file_t2_ax_seg}.nii.gz -discfile ${file_t2_ax_labels}.nii.gz -anat {file_t2_ax}.nii.gz -perslice 1 -o ${PATH_RESULTS}/T2w_ax_cord_metrics_perslice.csv -append 1

    # Normalized to PAM50 perlevel -- metrics across subjects are appended to a single CSV file
    sct_process_segmentation -i ${file_t2_ax_seg}.nii.gz -discfile ${file_t2_ax_labels}.nii.gz -anat {file_t2_ax}.nii.gz -normalize-PAM50 1 -perslice 1 -perlevel 1 -o ${PATH_RESULTS}/T2w_ax_cord_metrics_perlevel_PAM50.csv -append 1
    # Normalized to PAM50 perslice -- metrics across subjects are appended to a single CSV file
    sct_process_segmentation -i ${file_t2_ax_seg}.nii.gz -discfile ${file_t2_ax_labels}.nii.gz -anat {file_t2_ax}.nii.gz -normalize-PAM50 1 -perslice 1 -o ${PATH_RESULTS}/T2w_ax_cord_metrics_perslice_PAM50.csv -append 1

    # -------------
    # Segment spinal canal if manual segmentation doesn't exists
    # -------------
    segment_canal_if_does_not_exist ${file_t2_ax}
    file_t2_ax_canal_seg=$FILESEG

    # -------------
    # Compute spinal canal morphometrics
    # -------------
    echo "Computing spinal canal morphometrics..."
    # Compute canal metrics perlevel in the native space -- metrics across subjects are appended to a single CSV file
    sct_process_segmentation -i ${file_t2_ax_canal_seg}.nii.gz -discfile ${file_t2_ax_labels}.nii.gz -anat {file_t2_ax}.nii.gz -perlevel 1 -vert 2:9 -o ${PATH_RESULTS}/T2w_ax_canal_metrics_perlevel.csv -append 1
    # Compute canal metrics perslice in the native space -- metrics across subjects are appended to a single CSV file
    sct_process_segmentation -i ${file_t2_ax_canal_seg}.nii.gz -discfile ${file_t2_ax_labels}.nii.gz -anat {file_t2_ax}.nii.gz -perslice 1 -o ${PATH_RESULTS}/T2w_ax_canal_metrics_perslice.csv -append 1

    # Normalized to PAM50 perlevel -- metrics across subjects are appended to a single CSV file
    sct_process_segmentation -i ${file_t2_ax_canal_seg}.nii.gz -discfile ${file_t2_ax_labels}.nii.gz -anat {file_t2_ax}.nii.gz -normalize-PAM50 1 -perslice 1 -perlevel 1 -o ${PATH_RESULTS}/T2w_ax_canal_metrics_perlevel_PAM50.csv -append 1
    # Normalized to PAM50 perslice -- metrics across subjects are appended to a single CSV file
    sct_process_segmentation -i ${file_t2_ax_canal_seg}.nii.gz -discfile ${file_t2_ax_labels}.nii.gz -anat {file_t2_ax}.nii.gz -normalize-PAM50 1 -perslice 1 -o ${PATH_RESULTS}/T2w_ax_canal_metrics_perslice_PAM50.csv -append 1

    # -------------
    # Compute aSCOR -- it needs both SC and canal segmentations
    # -------------
    # Perlevel in the native space -- metrics across subjects are appended to a single CSV file
    sct_compute_ascor -i-SC ${file_t2_ax_seg}.nii.gz -i-canal ${file_t2_ax_canal_seg}.nii.gz -discfile ${file_t2_ax_labels}.nii.gz -perlevel 1 -vert 2:9 -o ${PATH_RESULTS}/T2w_ax_aSCOR_metrics_perlevel.csv -append 1
    # Perslice in the native space -- metrics across subjects are appended to a single CSV file
    sct_compute_ascor -i-SC ${file_t2_ax_seg}.nii.gz -i-canal ${file_t2_ax_canal_seg}.nii.gz -discfile ${file_t2_ax_labels}.nii.gz -perslice 1 -o ${PATH_RESULTS}/T2w_ax_aSCOR_metrics_perslice.csv -append 1

    # Normalized to PAM50 perlevel -- metrics across subjects are appended to a single CSV file
    sct_compute_ascor -i-SC ${file_t2_ax_seg}.nii.gz -i-canal ${file_t2_ax_canal_seg}.nii.gz -discfile ${file_t2_ax_labels}.nii.gz -normalize-PAM50 1 -perslice 1 -perlevel 1 -o ${PATH_RESULTS}/T2w_ax_aSCOR_metrics_perlevel_PAM50.csv -append 1
    # Normalized to PAM50 perslice -- metrics across subjects are appended to a single CSV
    sct_compute_ascor -i-SC ${file_t2_ax_seg}.nii.gz -i-canal ${file_t2_ax_canal_seg}.nii.gz -discfile ${file_t2_ax_labels}.nii.gz -normalize-PAM50 1 -perslice 1 -o ${PATH_RESULTS}/T2w_ax_aSCOR_metrics_perslice_PAM50.csv -append 1

#    # -------------
#    # Segment intramedullary lesions if manual segmentation doesn't exists
#    # -------------
#    segment_lesion_if_does_not_exist ${file_t2_ax} 't2'
#    file_t2_ax_lesion_seg=$FILESEG
#    # Compute lesion metrics
#    echo "Computing lesion metrics..."
#    # Check if there are any lesions by examining the segmentation file
#    # Use fslstats to check if there are non-zero voxels in the lesion segmentation
#    if command -v fslstats >/dev/null 2>&1; then
#        lesion_check=$(fslstats ${file_t2_ax_lesion_seg}_lesion_seg.nii.gz -V | awk '{print ($1 > 0) ? 1 : 0}')
#    else
#        # Fallback: check if file exists and has content
#        lesion_check=$(test -f ${file_t2_ax_lesion_seg}_lesion_seg.nii.gz && echo "1" || echo "0")
#    fi
#
#    if [[ $lesion_check -gt 0 ]]; then
#        echo "Found $lesion_check discrete lesion(s). Running lesion analysis..."
#        sct_analyze_lesion -m ${file_t2_ax_lesion_seg}_lesion_seg.nii.gz -s ${file_t2_ax_seg}.nii.gz -ofolder ${PATH_RESULTS}
#
#        # Use the reliable connected components count from SCT
#        # This is the most robust approach that works on all machines
#        lesion_objects_count=$lesion_check
#        echo "Lesion analysis complete. Found $lesion_objects_count discrete lesion(s)."
#    else
#        echo "No lesions found in segmentation. Skipping lesion analysis."
#        lesion_objects_count=0
#    fi

#    # -------------
#    # Create lesion and myelopathy summary
#    # -------------
#    echo "Creating lesion and myelopathy summary..."
#    SUMMARY_FILE="${PATH_RESULTS}/lesion_myelopathy_summary.csv"
#
#    # Create header if file doesn't exist
#    if [[ ! -f ${SUMMARY_FILE} ]]; then
#        echo "participant_id,lesion_count,myelopathy_count" > ${SUMMARY_FILE}
#    fi
#
#    # Get myelopathy count from participants.tsv
#    myelopathy_info=$(grep "^${SUBJECT}" ${PARTICIPANTS_PATH} | cut -f16)  # Assuming myelopathy is column 16
#    if [[ -n "$myelopathy_info" && "$myelopathy_info" != "n/a" ]]; then
#         Count myelopathies by counting commas and adding 1, or 0 if empty
#        myelopathy_count=$(echo "$myelopathy_info" | grep -o "," | wc -l)
#        myelopathy_count=$((myelopathy_count + 1))
#    else
#        myelopathy_count=0
#    fi
#
#    # Append data to summary file
#    echo "${SUBJECT},${lesion_objects_count},${myelopathy_count}" >> ${SUMMARY_FILE}
#    echo "Added to summary: ${SUBJECT} - Lesions: ${lesion_objects_count}, Myelopathies: ${myelopathy_count}"

#    # -------------
#    # Compute compression metrics
#    # -------------
#    # Check if file with compression labels exists.
#    file_compression="${file_t2_ax}_label-compression-manual"
#    FILE_COMPRESSION_MANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/${SESSION}/anat/${file_compression}.nii.gz"
#    if [[ ! -e ${FILE_COMPRESSION_MANUAL} ]]; then
#        echo "File ${FILE_COMPRESSION_MANUAL}.nii.gz does not exist" >> ${PATH_LOG}/missing_files.log
#        echo "ERROR: File ${FILE_COMPRESSION_MANUAL}.nii.gz does not exist. Exiting."
#        exit 1
#    else
#        echo "Found! Using manual compression labels."
#        rsync -avzh $FILE_COMPRESSION_MANUAL ${file_compression}.nii.gz
#
#        # Fetch sex from participants.tsv file (adaptated for new folder organization)
#        sex=$(grep ${SUBJECT} ${PARTICIPANTS_PATH} | awk '{print $4}')
#        echo "${SUBJECT}: ${sex}"
#
#        # TODO: test without angle correction too
#        # Compute morphometric measures normalized to PAM50 template space
#        # Note: CSV file without normalization is also generated automatically
#        # Note: morphometric measures for individual subjects are appended to a single CSV file
#        # diameter_AP
#        sct_compute_compression -i ${file_t2_ax_seg}.nii.gz -vertfile ${file_t2_ax_seg}_labeled.nii.gz -l ${file_compression}.nii.gz -normalize-hc 1 -sex ${sex} -o ${PATH_RESULTS}/compression_metrics.csv
#        # cross-sectional area
#        sct_compute_compression -i ${file_t2_ax_seg}.nii.gz -vertfile ${file_t2_ax_seg}_labeled.nii.gz -l ${file_compression}.nii.gz -normalize-hc 1 -sex ${sex} -metric area -o ${PATH_RESULTS}/compression_metrics.csv
#        # diameter_RL
#        sct_compute_compression -i ${file_t2_ax_seg}.nii.gz -vertfile ${file_t2_ax_seg}_labeled.nii.gz -l ${file_compression}.nii.gz -normalize-hc 1 -sex ${sex} -metric diameter_RL -o ${PATH_RESULTS}/compression_metrics.csv
#        # eccentricity
#        sct_compute_compression -i ${file_t2_ax_seg}.nii.gz -vertfile ${file_t2_ax_seg}_labeled.nii.gz -l ${file_compression}.nii.gz -normalize-hc 1 -sex ${sex} -metric eccentricity -o ${PATH_RESULTS}/compression_metrics.csv
#        # solidity
#        sct_compute_compression -i ${file_t2_ax_seg}.nii.gz -vertfile ${file_t2_ax_seg}_labeled.nii.gz -l ${file_compression}.nii.gz -normalize-hc 1 -sex ${sex} -metric solidity -o ${PATH_RESULTS}/compression_metrics.csv
#    fi

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
