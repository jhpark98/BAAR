# Packages
import time
import os
import pandas as pd
import numpy as np
import cv2
from ultralytics import YOLO

from helpers.Memory import MemoryStorage

import psutil

# Get the current process
process = psutil.Process()

# Get the CPU affinity (list of allowed cores)
cpu_affinity = process.cpu_affinity()

# Number of cores the process is using
# num_cores = len(cpu_affinity) # does not work in MacOS

print(f"The process is allowed to use {num_cores} core(s): {cpu_affinity}")


def load_image(fpath):
    return cv2.imread(fpath)

# TODO
def measure():
    pass

def gen_fpath(dir_path, lap_interval):
    path_lst = []
    for i in range(lap_interval[0], lap_interval[1]):
        fpath = os.path.join(dir_path, f"frame_{i}.png")
        path_lst.append(fpath)

    return path_lst

def handle_model_results(results):
    """
    Handle 0) one bounding box (our scenario for now) 1) no bounding box 2) multiple bounding boxes
    and return bbox_lst (a list of bbox)
    
    The input `results` is expected to be an object containing the detection results from the YOLO model.
    Each result in `results` contains bounding boxes and associated attributes.

    The structure of `results` is as follows:
    - results: List of result objects, each corresponding to an image.
        - result.boxes: List of bounding boxes detected in the image.
            - box.xyxy: Tensor containing bounding box coordinates in the format (x1, y1, x2, y2).
            - box.conf: Tensor containing the confidence score of the detection.
            - box.cls: Tensor containing the class ID of the detected object.

    Example:
    results[0].boxes
    ultrlalytics.engine.results.Boxes object with attributes:
        cls: tensor([11.])
        conf: tensor([0.4054])
        data: tensor([[1.9740e+03, 2.0553e+02, 2.0877e+03, 2.7732e+02, 4.0537e-01, 1.1000e+01]])
        id: None
        is_track: False
        orig_shape: (720, 2560)
        shape: torch.Size([1, 6])
        xywh: tensor([[2030.8615,  241.4213,  113.6453,   71.7887]])
        xywhn: tensor([[0.7933, 0.3353, 0.0444, 0.0997]])
        xyxy: tensor([[1974.0388,  205.5270, 2087.6841,  277.3157]])
        xyxyn: tensor([[0.7711, 0.2855, 0.8155, 0.3852]])
    """
    boxes = results[0].boxes
    bbox_lst = [tuple(map(int, box.xyxy[0].tolist())) for box in boxes]

    return bbox_lst

"----- ----- ----- ----- ----- ----- ----- -----"
# GLOBAL VARIABLES
GRID_SIZE = 10
model = YOLO("yolov8n.pt")

def main():
    frame_data = pd.read_csv("FrameandData.csv")
    
    dir_stopsign_1 = "Images_stopsign1"
    dir_stopsign_2 = "Images_stopsign2"

    stopsign_1_lap_1_interval = (730, 750)
    stopsign_1_lap_2_interval = (3485, 3630)
    stopsign_2_lap_1_interval = (2900, 3200)
    stopsign_2_lap_2_interval = (5550, 5722)

    frame_and_state = {}

    intervals = [stopsign_1_lap_1_interval, stopsign_1_lap_2_interval, stopsign_2_lap_1_interval, stopsign_2_lap_2_interval]
    for interval in intervals:
        for i in range(interval[0], interval[1]):
            row = frame_data.iloc[i-1]
            frame_and_state[i] = (np.float32(row['X_location(m)']), np.float32(row['Y_location(m)']), np.float32(row['Yaw_angle(deg)']))

    stopsign_1_lap_1 = gen_fpath(dir_stopsign_1, stopsign_1_lap_1_interval)
    stopsign_1_lap_2 = gen_fpath(dir_stopsign_1, stopsign_1_lap_2_interval)
    stopsign_2_lap_1 = gen_fpath(dir_stopsign_2, stopsign_2_lap_1_interval)
    stopsign_2_lap_2 = gen_fpath(dir_stopsign_2, stopsign_2_lap_2_interval)
    
    """
    object detector -> OD
    
    How the memory storage system works:
        1) LAP-1 — for each image frame (20 Hz)
            - 1) run the off-the-shelf OD, 2) create and save memory
        2) LAP-2-10 — for each image frame (20 Hz)
            - 1) query memory if memory exists
                1.1) if memory does not exist for the corresponding state, run OD & save a memory
                1.2) if memory exists, use the memory to 1) crop the image, 2) run OD, 3) update memory
                    1.2.1) TODO — UPDATE RULE
    """  
    
    print("Experiment start" + "\n"*5)
    LAP = 10
    for i in range(LAP):

        print(f"Processing Lap {i} ...")
        
        # Initialize memory
        memory = MemoryStorage()
        
        # lap #1 - build memory
        if i == 0: 
            
            for fpath in stopsign_1_lap_1:
                frame_number = int(fpath.split('_')[-1].split('.')[0])
                state_vector = frame_and_state[frame_number]

                # 1) run OD
                results = model(fpath, show=False, conf = 0.1)
                bbox_lst = handle_model_results(results)
                if len(bbox_lst) == 0:
                    print("No bounding box found in the current frame ...")
                    continue
                else:
                    # 2) create and save memory
                    memory_matrix = memory.create_memory_matrix(bbox_lst)
                    memory.add_memoery(state_vector, memory_matrix)
                time.sleep(1/20)  # Ensure processing each image is 20 Hz

        # lap #2-10 - utilize memory
        elif i > 3:
            break
        
        else:
            for fpath in stopsign_1_lap_2:
                org_img = cv2.imread(fpath)
                frame_number = int(fpath.split('_')[-1].split('.')[0])
                state_vector = frame_and_state[frame_number]
                
                memory_matrix = memory.query_memory(state_vector)
                if memory_matrix is None: # if no memory matrix, need to add a new matrix
                    # process whole image
                    results = model(fpath, show=False, conf = 0.1) 
                    bbox_lst = handle_model_results(results)
                    
                    # if bbox_lst
                    memory_matrix = memory.create_memory_matrix(bbox_lst)
                    memory.add_memoery(state_vector, memory_matrix)
                else:
                    # process partial image with memory filter
                    results = process_img_with_memory(org_img, memory_matrix)

                    # update memory matrix with results

def process_img_with_memory(org_img, memory_matrix, GRID_SIZE=10):
    ''''
    Use memory filter to filter the region of interest and apply OD to the filterd/cropped images
    '''

    n, m, _ = org_img.shape
    filtered_img = None

    for i in range(memory_matrix.shape[0]):
        for j in range(memory_matrix.shape[1]):
            if memory_matrix[i, j] == 1:
                x_start = j * GRID_SIZE
                y_start = i * GRID_SIZE
                x_end = min((j + 1) * GRID_SIZE, m)
                y_end = min((i + 1) * GRID_SIZE, n)
                filtered_img = org_img[y_start:y_end, x_start:x_end]

    results = model(filtered_img, show=False, conf=0.1)
    return handle_model_results(results)

if __name__ == "__main__":
    main()