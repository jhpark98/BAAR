import pathlib
import glob
import os
import sys
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt

def overlay_heatmap_on_image(image, heatmap):
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min()) # Normalize the heatmap
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))       # Resize heatmap to match image size
    heatmap = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET) # Apply color map to heatmap
    overlaid_image = cv2.addWeighted(image, 0.5, heatmap, 0.5, 0) # Overlay heatmap on image
    return overlaid_image