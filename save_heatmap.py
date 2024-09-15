import pathlib
import glob
import os
import sys
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt

from helpers.Vehicle import Vehicle
from helpers.Gaze import Gaze
# from helpers.helper import *

def overlay_heatmap_on_image(image, heatmap):
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min()) # Normalize the heatmap
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))       # Resize heatmap to match image size
    heatmap = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET) # Apply color map to heatmap
    overlaid_image = cv2.addWeighted(image, 0.5, heatmap, 0.5, 0) # Overlay heatmap on image
    return overlaid_image


# SET UP TESTING ENVIRONMENT

RUN_IDX = 6 # e.g., run_{RUN_IDX}
N_LAPS = 10

SAVE_DIR = pathlib.Path(".").joinpath(f"run_{RUN_IDX}").absolute()
GAZE_DIR = pathlib.Path(max(glob.glob(os.path.join(SAVE_DIR, '*/')), key=os.path.getmtime))

PATH_GAZE = GAZE_DIR.joinpath("gaze_positions.csv")
PATH_BLINK = GAZE_DIR.joinpath("blinks.csv")
PATH_VEHICLE = SAVE_DIR.joinpath("df_vehicle.csv")

gaze = Gaze(PATH_GAZE, PATH_BLINK)
vehicle = Vehicle(PATH_VEHICLE, N_LAPS)
FrameandData = pd.read_csv("FrameandData.csv")
print("TESTING ENVIRONMENT SET UP DONE.")

print("----- " * 5)

print("Grab data for specific section with condition ...")
section_key = "stop_1"
veh_ts = vehicle.get_ts(section_key, 5.0) # e.g., get all the timestamps, 5.0 m before the full stop
print(f"Processing gaze maps for {section_key}")
gaze_ts, gaze_idx = gaze.veh_ts_to_gaze_ts(veh_ts) # grab syncrhonized gaze data
heatmap_lst = gaze.gen_gaze_map(gaze_idx)          # generate gaze map

# grab syncrhonized frame
closest_frame_idx_lst = [(FrameandData['real_time(s)'] - gaze_ts_lap).abs().idxmin() for gaze_ts_lap in gaze_ts]
frame_ts = [FrameandData.loc[idx, 'real_time(s)'] for idx in closest_frame_idx_lst]  # Assign frame timestamps
img_path_lst = [os.path.join("Images_stopsign1", f"frame_{FrameandData.loc[idx, 'Frame']}.png") for idx in closest_frame_idx_lst]

print("Calculating differences between veh_ts, gaze_ts, and frame_ts ...")
for i, (v_ts, g_ts, f_ts) in enumerate(zip(veh_ts, gaze_ts, frame_ts)):
    veh_gaze_diff = v_ts - g_ts
    gaze_frame_diff = g_ts - f_ts
    veh_frame_diff = v_ts - f_ts
    print(f"Index {i}: veh_ts - gaze_ts = {veh_gaze_diff}, gaze_ts - frame_ts = {gaze_frame_diff}, veh_ts - frame_ts = {veh_frame_diff}")
    
print("\n----- ----- Beginning ----- -----")
for lap, (gaze_ts_lap, heatmap_lap, img_path) in enumerate(zip(gaze_ts, heatmap_lst, img_path_lst)):
    if lap > 6:
        continue

    print(f"\nProcessing lap {lap + 1}/{N_LAPS}")

    print(f"Reading in {img_path} ...")
    image_frame = cv2.imread(img_path)
    print(f"Size of the image: {image_frame.shape}")
    image_frame = cv2.resize(image_frame, (image_frame.shape[1] // 2, image_frame.shape[0] // 2)) # (1080, 360) 

    # grab (2N + 1) gaze data
    windowed_gaze_data = gaze.grab_window(gaze_ts_lap)
    scaled_gaze = (np.array(windowed_gaze_data[['norm_pos_x', 'norm_pos_y']]) * np.array([image_frame.shape[1], image_frame.shape[0]])).astype(int)
    for gaze_point in scaled_gaze:
        cv2.circle(image_frame, tuple(gaze_point), radius=5, color=(0, 0, 255), thickness=-1)

    # Visualize the image using cv2
    cv2.imshow(f"Overlaid Image Lap {lap + 1}", image_frame)
    cv2.waitKey(0)  # Wait for a key press to close the image window
    cv2.destroyAllWindows()

    # GAZE HEATMAP
    overlaid_image = overlay_heatmap_on_image(image_frame, heatmap_lap)
    # Saving overlaid images
    save_dir = "overlaid"
    image_frame_number = int(os.path.basename(img_path).split('_')[1].split('.')[0])
    output_path = f"overlaid_frame_{image_frame_number}.png"
    output_path = os.path.join(save_dir, output_path)
    cv2.imwrite(output_path, overlaid_image)
    print(f"Saved overlaid image to {output_path}")

sys.exit()