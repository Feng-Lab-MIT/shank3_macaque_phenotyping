import numpy as np
import pandas as pd
from tqdm import tqdm
import glob
import cv2
import pickle

# Extract video clips for each cluster to see the behaviors behind it
output_folder = '/media/fenglab/Seagate Hub/Shank3_macaque/second cohort/social/limb_model/relative_features/video_chunks/200_bp_ba_seq_240_pred_24_32_clusters_normal_features/'
#output_folder = '/media/fenglab/Seagate Hub/Shank3_macaque/second cohort/social/video_chunks/second_cohort_two_attention/'
video_folder = '/mnt/Second_Cohort_Social/'
labels = pd.read_csv('/home/fenglab/Documents/dlc/biomarker_model/second_cohort_labels.csv', header=[0], dtype=str)


sequence_length = 240  # Changed from 480 to 48
prediction_length = 24  # Changed from 48 to 24
original_sequence_length = 480  # The sequence length used when selecting category data

# Calculate the offset needed to adjust starting positions
sequence_offset = original_sequence_length - sequence_length
print(f"Sequence offset adjustment: {sequence_offset} frames")

#tsne_file = '/media/newhd/River_data_backup/2nd_cohort_social/3d/model/tsne_results_sequence_length_480_output_feature_18_perplexity_125_exaggeration_1_interval_8_earlyiter_300_pca_1478_n_neighbors_50_1_5.csv'
#tsne_file = '/media/fenglab/Seagate Hub/Shank3_macaque/second cohort/social/tsne_results_sequence_length_480_output_feature_18_perplexity_200_exaggeration_1_interval_8_earlyiter_300_pca_1676_n_neighbors_50.csv'
tsne_file = '/media/fenglab/newssd/social/bottom_playing_and_alone_pca_seq_240_latent_512_bottleneck_256_no_velocity_test_depth/tsne_results_subset_Bottom_Playing_and_Alone_perplexity_200_exaggeration_1_earlyiter_300_pca_533_n_neighbors_50_n_batches_6_n_clusters_32_no_velocity_test_depth_sequence_length_240.csv'
tsne_df = pd.read_csv(tsne_file)

with open(f'/media/fenglab/New Volume/River_data_backup/2nd_cohort_social/3d/model/1_5/stage_dic_1_5.pkl','rb') as f:
    stage_dic = pickle.load(f)
tsne_df['Stage'] = tsne_df['DataFrame Index'].map(stage_dic)

def time_to_frame(ts, fps):
    hour = ts.rsplit(":", 2)[0]
    minute = ts.rsplit(":", 2)[1]
    second = ts.rsplit(":", 2)[2]
    total_second = int(hour)*3600 + int(minute)*60 + int(second)
    return total_second*fps

def seconds_to_mmss(seconds):
    minutes = seconds // 60
    secs = seconds % 60
    return f"{minutes:02}:{secs:02}"
fps = 24

def extract_videos(tsne_df, output_folder, labels, video_folder, subsample=None, candidate_list=None, monkey_list=None):
    for cluster in range(128):
        print(cluster)
        if candidate_list and cluster not in candidate_list:
            continue
        cluster_idx = tsne_df.index[tsne_df['Cluster'] == cluster].tolist()
        if len(cluster_idx) == 0:
            print("Cluster Info Not Available.")
            continue
        clip = 0
        if subsample:
            print(cluster_idx)
            if subsample > 1:
                if len(cluster_idx) > subsample:
                    rng = np.random.default_rng()
                    cluster_idx = rng.choice(
                        cluster_idx, size=subsample, replace=False)
            else:
                subsample_size = int(subsample*len(cluster_idx))
                cluster_idx = np.random.choice(
                    cluster_idx, size=subsample_size, replace=False)
        for idx in cluster_idx:
            # print(idx)
            monkey_day = tsne_df.loc[idx, 'Monkey_Day']
            #if monkey_day in ['asd_0927','asd_0929','sdf_0927','sdf_0929']:
            #    continue
            print(monkey_day)
            monkey = monkey_day.rsplit('_', 1)[0]
            if monkey_list and monkey not in monkey_list:
                continue
            day = monkey_day.rsplit('_', 1)[1]
            session = 0
            stage = tsne_df.loc[idx, 'Stage']
            date = labels.loc[(labels['Stage'] == stage) & (labels['Test'] == monkey) & ((labels['Day'] == day)|('0'+labels['Day'] == day)) & (
                labels['Session'] == str(session)), 'Date'].iloc[0]
            session_start = labels.loc[(labels['Stage'] == stage) & (labels['Test'] == monkey) & ((labels['Day'] == day)|('0'+labels['Day'] == day)) & (
                labels['Session'] == str(session)), 'session_start'].iloc[0]
            #stage = labels.loc[(labels['Test'] == monkey) & ((labels['Day'] == day)|('0'+labels['Day'] == day)) & (
            #    labels['Session'] == str(session)), 'Stage'].iloc[0].lower()
            if session_start == 'Y':
                session_start = "00:01:00"
            session_start_frame = time_to_frame(session_start, fps)
            new_video_folder = f'{video_folder}{stage}/{monkey}/full/'
            video_name = new_video_folder+monkey + \
                '_cam1-'+date[-4:]+date[:4]+'*.avi'
            print(video_name)
            matching = glob.glob(video_name)
            #start_frame = tsne_df.loc[idx,'Frame Index']+session_start_frame+(sequence_length-24) #start frame is 1 second before predictiong window
            #end_frame = start_frame + (prediction_length + 24) #end frame is end of prediction window
            start_frame = tsne_df.loc[idx,'Frame Index']+session_start_frame
            # Apply sequence offset adjustment
            start_frame = start_frame + sequence_offset
            highlight_frame = start_frame + sequence_length - 1
            end_frame = highlight_frame + prediction_length
            if matching:
                print("Video name found")
                if len(matching) > 1:
                    print("More than 1 video found, skipping...")
                    continue
                for file in matching:
                    print(file)
                    cap = cv2.VideoCapture(file)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                    output_path = f'{output_folder}{cluster}_{monkey_day}_{clip}_{start_frame}.mp4'
                    #print(output_path)
                    output_fps = cap.get(cv2.CAP_PROP_FPS)
                    #print(output_fps)
                    output_codec = cv2.VideoWriter_fourcc(*'mp4v')
                    output_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
                        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                    out = cv2.VideoWriter(
                        output_path, output_codec, output_fps, output_size)
                    # Set current frame to start frame
                    # Iterate through frames and write to output video
                    for i in range(start_frame, end_frame):
                        # Read next frame
                        ret, frame = cap.read()
                        # Check if frame was successfully read
                        if not ret:
                            break
                        if i == highlight_frame:
                            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 10)
                        # Write frame to output video
                        _ = out.write(frame)
                    # Release video capture and output video objects
                    cap.release()
                    out.release()
                    clip += 1
            else:
                print("Video name not found")

#monkey_list = ['asd']
#candidate_list = [36,111]
extract_videos(tsne_df, output_folder, labels, video_folder, subsample=20)