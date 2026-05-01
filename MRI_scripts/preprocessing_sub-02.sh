#!/bin/bash

# ====================
# sub-02 setup and pre-afni_proc inputs
# ====================

# ===== HISTORY source: sub-02_ses-01_task-rest_run-1_bold.ts.fm.deo.ref.res.cut.nii =====
to3d -prefix sub-02_ses-01_task-rest_run-1_bold.nii.gz -time:zt 38 128 2.8sec FROM_IMAGE CF57/CF57_1_6_20210107_ep2d_bold_NEW_1/05518733.dcm CF57/CF57_1_6_20210107_ep2d_bold_NEW_1/05518717.dcm CF57/CF57_1_6_20210107_ep2d_bold_NEW_1/05518701.dcm ... CF57/CF57_1_6_20210107_ep2d_bold_NEW_1/05519968.dcm CF57/CF57_1_6_20210107_ep2d_bold_NEW_1/05519952.dcm
3dTshift -tzero 0 -prefix sub-02_ses-01_task-rest_run-1_bold.ts.nii.gz sub-02_ses-01_task-rest_run-1_bold.nii.gz
3dcalc -a sub-02_ses-01_task-rest_run-1_bold.ts.nii.gz -expr a -prefix ./__work_B0_corr_ZbnpWueDoHj/sub-02_ses-01_task-rest_run-1_bold.ts.nii.gz
3dNwarpApply -warp sub-02_ses-01_task-rest_run-1_bold.ts.fm_WARP.nii.gz -prefix sub-02_ses-01_task-rest_run-1_bold.ts.fm_EPI.nii.gz -source sub-02_ses-01_task-rest_run-1_bold.ts.nii.gz
3dWarp -deoblique -prefix __sub-02_ses-01_task-rest_run-1_bold.ts.fm.deo.nii.gz __sub-02_ses-01_task-rest_run-1_bold.ts.fm.nii.gz
3dresample -orient LPI -prefix sub-02_ses-01_task-rest_run-1_bold.ts.fm.deo.ref.res.nii.gz -input __sub-02_ses-01_task-rest_run-1_bold.ts.fm.deo.ref.nii.gz
3dZeropad -mm -S -14.9853 -I -16.0147 -P -12.855 -A -16.1449 -L -14.535 -R -16.465 -prefix sub-02_ses-01_task-rest_run-1_bold.ts.fm.deo.ref.res.cut.nii analysis_dimon/derivatives/first_level/sub-02/ses-01/reco/sub-02_ses-01_task-rest_run-1_bold.ts.fm.deo.ref.res.nii
# ===== HISTORY source: sub-02_ses-01_task-rest_run-2_bold.ts.fm.deo.ref.res.cut.nii =====
to3d -prefix sub-02_ses-01_task-rest_run-2_bold.nii.gz -time:zt 38 128 2.8sec FROM_IMAGE CF57/CF57_1_7_20210107_ep2d_bold_NEW_2/05517530.dcm CF57/CF57_1_7_20210107_ep2d_bold_NEW_2/05517514.dcm CF57/CF57_1_7_20210107_ep2d_bold_NEW_2/05517498.dcm ... CF57/CF57_1_7_20210107_ep2d_bold_NEW_2/05518765.dcm CF57/CF57_1_7_20210107_ep2d_bold_NEW_2/05518749.dcm
3dTshift -tzero 0 -prefix sub-02_ses-01_task-rest_run-2_bold.ts.nii.gz sub-02_ses-01_task-rest_run-2_bold.nii.gz
3dcalc -a sub-02_ses-01_task-rest_run-2_bold.ts.nii.gz -expr a -prefix ./__work_B0_corr_lvb7au8eVQo/sub-02_ses-01_task-rest_run-2_bold.ts.nii.gz
3dNwarpApply -warp sub-02_ses-01_task-rest_run-2_bold.ts.fm_WARP.nii.gz -prefix sub-02_ses-01_task-rest_run-2_bold.ts.fm_EPI.nii.gz -source sub-02_ses-01_task-rest_run-2_bold.ts.nii.gz
3dWarp -deoblique -prefix __sub-02_ses-01_task-rest_run-2_bold.ts.fm.deo.nii.gz __sub-02_ses-01_task-rest_run-2_bold.ts.fm.nii.gz
3dresample -orient LPI -prefix sub-02_ses-01_task-rest_run-2_bold.ts.fm.deo.ref.res.nii.gz -input __sub-02_ses-01_task-rest_run-2_bold.ts.fm.deo.ref.nii.gz
3dZeropad -mm -S -14.9853 -I -16.0147 -P -12.855 -A -16.1449 -L -14.535 -R -16.465 -prefix sub-02_ses-01_task-rest_run-2_bold.ts.fm.deo.ref.res.cut.nii analysis_dimon/derivatives/first_level/sub-02/ses-01/reco/sub-02_ses-01_task-rest_run-2_bold.ts.fm.deo.ref.res.nii
# ===== HISTORY source: sub-02_ses-01_task-rest_run-3_bold.ts.fm.deo.ref.res.cut.nii =====
to3d -prefix sub-02_ses-01_task-rest_run-3_bold.nii.gz -time:zt 38 128 2.8sec FROM_IMAGE CF57/CF57_1_8_20210107_ep2d_bold_NEW_3/05513093.dcm CF57/CF57_1_8_20210107_ep2d_bold_NEW_3/05513077.dcm CF57/CF57_1_8_20210107_ep2d_bold_NEW_3/05513061.dcm ... CF57/CF57_1_8_20210107_ep2d_bold_NEW_3/05517562.dcm CF57/CF57_1_8_20210107_ep2d_bold_NEW_3/05517546.dcm
3dTshift -tzero 0 -prefix sub-02_ses-01_task-rest_run-3_bold.ts.nii.gz sub-02_ses-01_task-rest_run-3_bold.nii.gz
3dcalc -a sub-02_ses-01_task-rest_run-3_bold.ts.nii.gz -expr a -prefix ./__work_B0_corr_dVR7gZtOQJA/sub-02_ses-01_task-rest_run-3_bold.ts.nii.gz
3dNwarpApply -warp sub-02_ses-01_task-rest_run-3_bold.ts.fm_WARP.nii.gz -prefix sub-02_ses-01_task-rest_run-3_bold.ts.fm_EPI.nii.gz -source sub-02_ses-01_task-rest_run-3_bold.ts.nii.gz
3dWarp -deoblique -prefix __sub-02_ses-01_task-rest_run-3_bold.ts.fm.deo.nii.gz __sub-02_ses-01_task-rest_run-3_bold.ts.fm.nii.gz
3dresample -orient LPI -prefix sub-02_ses-01_task-rest_run-3_bold.ts.fm.deo.ref.res.nii.gz -input __sub-02_ses-01_task-rest_run-3_bold.ts.fm.deo.ref.nii.gz
3dZeropad -mm -S -14.9853 -I -16.0147 -P -12.855 -A -16.1449 -L -14.535 -R -16.465 -prefix sub-02_ses-01_task-rest_run-3_bold.ts.fm.deo.ref.res.cut.nii analysis_dimon/derivatives/first_level/sub-02/ses-01/reco/sub-02_ses-01_task-rest_run-3_bold.ts.fm.deo.ref.res.nii

# ====================
# sub-02 run-1 initial afni_proc commands
# ====================

# ===== HISTORY source: pb00.sub-02_ses-01.r01.tcat+orig =====
3dTcat -prefix sub-02_ses-01.results/pb00.sub-02_ses-01.r01.tcat 'sub-02_ses-01_task-rest_run-1_bold.ts.fm.deo.ref.res.cut.nii[0..$]'
# ===== HISTORY source: pb04.sub-02_ses-01.r01.scale+tlrc.HEAD =====
3dDespike -NEW -nomask -prefix pb01.sub-02_ses-01.r01.despike pb00.sub-02_ses-01.r01.tcat+orig
3dNwarpApply -master sub-05_ses-01_T1w.cut_warp2std_nsu+tlrc -dxyz 1.5 -source pb01.sub-02_ses-01.r01.despike+orig -nwarp 'sub-05_ses-01_T1w.cut_shft_WARP.nii.gz                              sub-05_ses-01_T1w.cut_composite_linear_to_template.1D sub-05_ses-01_aff12_inv.aff12.1D sub-02_ses-01_vr_base_min_outlier.lu.warped2base_WARP.nii.gz sub-02_ses-01_vr_base_min_outlier.lu_shft_al2base_mat.aff12.1D sub-02_ses-01_vr_base_min_outlier.lu_shft.1D mat.r01.vr.aff12.1D' -prefix rm.epi.nomask.r01
3dcalc -a pb01.sub-02_ses-01.r01.despike+orig -expr 1 -prefix rm.epi.all1
3dNwarpApply -master sub-05_ses-01_T1w.cut_warp2std_nsu+tlrc -dxyz 1.5 -source rm.epi.all1+orig -nwarp 'sub-05_ses-01_T1w.cut_shft_WARP.nii.gz                              sub-05_ses-01_T1w.cut_composite_linear_to_template.1D sub-05_ses-01_aff12_inv.aff12.1D sub-02_ses-01_vr_base_min_outlier.lu.warped2base_WARP.nii.gz sub-02_ses-01_vr_base_min_outlier.lu_shft_al2base_mat.aff12.1D sub-02_ses-01_vr_base_min_outlier.lu_shft.1D mat.r01.vr.aff12.1D' -interp cubic -ainterp NN -quiet -prefix rm.epi.1.r01
3dTstat -min -prefix rm.epi.min.r01 rm.epi.1.r01+tlrc

# ====================
# sub-02 run-2 initial afni_proc commands
# ====================

# ===== HISTORY source: pb00.sub-02_ses-01.r02.tcat+orig =====
3dTcat -prefix sub-02_ses-01.results/pb00.sub-02_ses-01.r02.tcat 'sub-02_ses-01_task-rest_run-2_bold.ts.fm.deo.ref.res.cut.nii[0..$]'
# ===== HISTORY source: pb04.sub-02_ses-01.r02.scale+tlrc.HEAD =====
3dDespike -NEW -nomask -prefix pb01.sub-02_ses-01.r02.despike pb00.sub-02_ses-01.r02.tcat+orig
3dNwarpApply -master sub-05_ses-01_T1w.cut_warp2std_nsu+tlrc -dxyz 1.5 -source pb01.sub-02_ses-01.r02.despike+orig -nwarp 'sub-05_ses-01_T1w.cut_shft_WARP.nii.gz                              sub-05_ses-01_T1w.cut_composite_linear_to_template.1D sub-05_ses-01_aff12_inv.aff12.1D sub-02_ses-01_vr_base_min_outlier.lu.warped2base_WARP.nii.gz sub-02_ses-01_vr_base_min_outlier.lu_shft_al2base_mat.aff12.1D sub-02_ses-01_vr_base_min_outlier.lu_shft.1D mat.r02.vr.aff12.1D' -prefix rm.epi.nomask.r02
# ===== HISTORY source: supplemental afni_proc log dependency =====
3dNwarpApply -master sub-05_ses-01_T1w.cut_warp2std_nsu+tlrc -dxyz 1.5 -source rm.epi.all1+orig -nwarp sub-05_ses-01_T1w.cut_shft_WARP.nii.gz sub-05_ses-01_T1w.cut_composite_linear_to_template.1D sub-05_ses-01_aff12_inv.aff12.1D sub-02_ses-01_vr_base_min_outlier.lu.warped2base_WARP.nii.gz sub-02_ses-01_vr_base_min_outlier.lu_shft_al2base_mat.aff12.1D sub-02_ses-01_vr_base_min_outlier.lu_shft.1D mat.r02.vr.aff12.1D -interp cubic -ainterp NN -quiet -prefix rm.epi.1.r02
3dTstat -min -prefix rm.epi.min.r02 rm.epi.1.r02+tlrc

# ====================
# sub-02 run-3 initial afni_proc commands
# ====================

# ===== HISTORY source: pb00.sub-02_ses-01.r03.tcat+orig =====
3dTcat -prefix sub-02_ses-01.results/pb00.sub-02_ses-01.r03.tcat 'sub-02_ses-01_task-rest_run-3_bold.ts.fm.deo.ref.res.cut.nii[0..$]'
# ===== HISTORY source: pb04.sub-02_ses-01.r03.scale+tlrc.HEAD =====
3dDespike -NEW -nomask -prefix pb01.sub-02_ses-01.r03.despike pb00.sub-02_ses-01.r03.tcat+orig
3dNwarpApply -master sub-05_ses-01_T1w.cut_warp2std_nsu+tlrc -dxyz 1.5 -source pb01.sub-02_ses-01.r03.despike+orig -nwarp 'sub-05_ses-01_T1w.cut_shft_WARP.nii.gz                              sub-05_ses-01_T1w.cut_composite_linear_to_template.1D sub-05_ses-01_aff12_inv.aff12.1D sub-02_ses-01_vr_base_min_outlier.lu.warped2base_WARP.nii.gz sub-02_ses-01_vr_base_min_outlier.lu_shft_al2base_mat.aff12.1D sub-02_ses-01_vr_base_min_outlier.lu_shft.1D mat.r03.vr.aff12.1D' -prefix rm.epi.nomask.r03
# ===== HISTORY source: supplemental afni_proc log dependency =====
3dNwarpApply -master sub-05_ses-01_T1w.cut_warp2std_nsu+tlrc -dxyz 1.5 -source rm.epi.all1+orig -nwarp sub-05_ses-01_T1w.cut_shft_WARP.nii.gz sub-05_ses-01_T1w.cut_composite_linear_to_template.1D sub-05_ses-01_aff12_inv.aff12.1D sub-02_ses-01_vr_base_min_outlier.lu.warped2base_WARP.nii.gz sub-02_ses-01_vr_base_min_outlier.lu_shft_al2base_mat.aff12.1D sub-02_ses-01_vr_base_min_outlier.lu_shft.1D mat.r03.vr.aff12.1D -interp cubic -ainterp NN -quiet -prefix rm.epi.1.r03
3dTstat -min -prefix rm.epi.min.r03 rm.epi.1.r03+tlrc

# ====================
# sub-02 shared EPI mask commands
# ====================

# ===== HISTORY source: pb04.sub-02_ses-01.r01.scale+tlrc.HEAD =====
3dMean -datum short -prefix rm.epi.mean rm.epi.min.r01+tlrc.HEAD rm.epi.min.r02+tlrc.HEAD rm.epi.min.r03+tlrc.HEAD
3dcalc -a rm.epi.mean+tlrc -expr 'step(a-0.999)' -prefix mask_epi_extents

# ====================
# sub-02 run-1 scale commands
# ====================

# ===== HISTORY source: pb04.sub-02_ses-01.r01.scale+tlrc.HEAD =====
3dcalc -a rm.epi.nomask.r01+tlrc -b mask_epi_extents+tlrc -expr 'a*b' -prefix pb02.sub-02_ses-01.r01.volreg
3dmerge -1blur_fwhm 6 -doall -prefix pb03.sub-02_ses-01.r01.blur pb02.sub-02_ses-01.r01.volreg+tlrc
3dTstat -prefix rm.mean_r01 pb03.sub-02_ses-01.r01.blur+tlrc
3dcalc -a pb03.sub-02_ses-01.r01.blur+tlrc -b rm.mean_r01+tlrc -c mask_epi_extents+tlrc -expr 'c * min(200, a/b*100)*step(a)*step(b)' -prefix pb04.sub-02_ses-01.r01.scale

# ====================
# sub-02 run-2 scale commands
# ====================

# ===== HISTORY source: pb04.sub-02_ses-01.r02.scale+tlrc.HEAD =====
3dcalc -a rm.epi.nomask.r02+tlrc -b mask_epi_extents+tlrc -expr 'a*b' -prefix pb02.sub-02_ses-01.r02.volreg
3dmerge -1blur_fwhm 6 -doall -prefix pb03.sub-02_ses-01.r02.blur pb02.sub-02_ses-01.r02.volreg+tlrc
3dTstat -prefix rm.mean_r02 pb03.sub-02_ses-01.r02.blur+tlrc
3dcalc -a pb03.sub-02_ses-01.r02.blur+tlrc -b rm.mean_r02+tlrc -c mask_epi_extents+tlrc -expr 'c * min(200, a/b*100)*step(a)*step(b)' -prefix pb04.sub-02_ses-01.r02.scale

# ====================
# sub-02 run-3 scale commands
# ====================

# ===== HISTORY source: pb04.sub-02_ses-01.r03.scale+tlrc.HEAD =====
3dcalc -a rm.epi.nomask.r03+tlrc -b mask_epi_extents+tlrc -expr 'a*b' -prefix pb02.sub-02_ses-01.r03.volreg
3dmerge -1blur_fwhm 6 -doall -prefix pb03.sub-02_ses-01.r03.blur pb02.sub-02_ses-01.r03.volreg+tlrc
3dTstat -prefix rm.mean_r03 pb03.sub-02_ses-01.r03.blur+tlrc
3dcalc -a pb03.sub-02_ses-01.r03.blur+tlrc -b rm.mean_r03+tlrc -c mask_epi_extents+tlrc -expr 'c * min(200, a/b*100)*step(a)*step(b)' -prefix pb04.sub-02_ses-01.r03.scale

# ====================
# sub-02 final all-runs regression command
# ====================

# ===== HISTORY source: errts.sub-02_ses-01.tproject+tlrc =====
3dTproject -polort 0 -input pb04.sub-02_ses-01.r01.scale+tlrc.HEAD pb04.sub-02_ses-01.r02.scale+tlrc.HEAD pb04.sub-02_ses-01.r03.scale+tlrc.HEAD -censor censor_sub-02_ses-01_combined_2.1D -cenmode ZERO -ort X.nocensor.xmat.1D -prefix errts.sub-02_ses-01.tproject
