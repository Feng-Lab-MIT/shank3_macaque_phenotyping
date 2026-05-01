#!/bin/bash

for sub in $(seq -w 01 15); do

    3dcopy \
        "errts.sub-${sub}_ses-01.tproject" \
        "errts.sub-${sub}_ses-01.tproject.nii.gz"

    3dTcorrMap \
        -input "errts.sub-${sub}_ses-01.tproject.nii.gz" \
        -polort -1 \
        -mask "sub-${sub}_ses-01.GM_mask_intersect.nii.gz" \
        -Mean "fc_tcorrmap.sub-${sub}_ses-01.blur_6.0mm.mean.nii.gz" \
        -Zmean "fc_tcorrmap.sub-${sub}_ses-01.blur_6.0mm.zmean.nii.gz"

done

3dttest++ -Clustsim -setA fc_tcorrmap.sub-01_ses-01.blur_6.0mm.zmean.nii.gz fc_tcorrmap.sub-02_ses-01.blur_6.0mm.zmean.nii.gz fc_tcorrmap.sub-03_ses-01.blur_6.0mm.zmean.nii.gz fc_tcorrmap.sub-04_ses-01.blur_6.0mm.zmean.nii.gz fc_tcorrmap.sub-05_ses-01.blur_6.0mm.zmean.nii.gz fc_tcorrmap.sub-06_ses-01.blur_6.0mm.zmean.nii.gz fc_tcorrmap.sub-07_ses-01.blur_6.0mm.zmean.nii.gz fc_tcorrmap.sub-08_ses-01.blur_6.0mm.zmean.nii.gz -setB fc_tcorrmap.sub-09_ses-01.blur_6.0mm.zmean.nii.gz fc_tcorrmap.sub-10_ses-01.blur_6.0mm.zmean.nii.gz fc_tcorrmap.sub-11_ses-01.blur_6.0mm.zmean.nii.gz fc_tcorrmap.sub-12_ses-01.blur_6.0mm.zmean.nii.gz fc_tcorrmap.sub-13_ses-01.blur_6.0mm.zmean.nii.gz fc_tcorrmap.sub-14_ses-01.blur_6.0mm.zmean.nii.gz fc_tcorrmap.sub-15_ses-01.blur_6.0mm.zmean.nii.gz -AminusB -no1sam -zskip 5 -toz -mask Cyno162-GM_mask-0p5.rs.nii -prefix ttest.fc_tcorrmap.groups_WT_S3.blur_6.0mm_bandpass.nii.gz
