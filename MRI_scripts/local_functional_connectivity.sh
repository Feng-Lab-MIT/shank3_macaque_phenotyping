#!/bin/bash

for sub in $(seq -w 01 15); do

    3dcopy \
        "errts.sub-${sub}_ses-01.tproject" \
        "errts.sub-${sub}_ses-01.tproject.nii"

    3dLFCD \
        -faces_edges \
        -prefix "fc_lfcd.sub-${sub}_ses-01.blur_6.0mm.nii.gz" \
        -polort -1 \
        -thresh 0.6 \
        -mask "sub-${sub}_ses-01.GM_mask_intersect.nii.gz" \
        "errts.sub-${sub}_ses-01.tproject.nii"

    3dTcat \
        -prefix "fc_lfcd.sub-${sub}_ses-01.blur_6.0mm.voxcount.nii.gz" \
        "fc_lfcd.sub-${sub}_ses-01.blur_6.0mm.nii.gz[0]"

done

3dttest++ -Clustsim -setA fc_lfcd.sub-01_ses-01.blur_6.0mm.voxcount.nii.gz fc_lfcd.sub-02_ses-01.blur_6.0mm.voxcount.nii.gz fc_lfcd.sub-03_ses-01.blur_6.0mm.voxcount.nii.gz fc_lfcd.sub-04_ses-01.blur_6.0mm.voxcount.nii.gz fc_lfcd.sub-05_ses-01.blur_6.0mm.voxcount.nii.gz fc_lfcd.sub-06_ses-01.blur_6.0mm.voxcount.nii.gz fc_lfcd.sub-07_ses-01.blur_6.0mm.voxcount.nii.gz fc_lfcd.sub-08_ses-01.blur_6.0mm.voxcount.nii.gz -setB fc_lfcd.sub-09_ses-01.blur_6.0mm.voxcount.nii.gz fc_lfcd.sub-10_ses-01.blur_6.0mm.voxcount.nii.gz fc_lfcd.sub-11_ses-01.blur_6.0mm.voxcount.nii.gz fc_lfcd.sub-12_ses-01.blur_6.0mm.voxcount.nii.gz fc_lfcd.sub-13_ses-01.blur_6.0mm.voxcount.nii.gz fc_lfcd.sub-14_ses-01.blur_6.0mm.voxcount.nii.gz fc_lfcd.sub-15_ses-01.blur_6.0mm.voxcount.nii.gz -AminusB -no1sam -zskip 5 -toz -mask Cyno162-GM_mask-0p5.rs.nii -prefix ttest.fc_lfcd.groups_WT_S3.blur_6.0mm_bandpass.nii.gz

