import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter # heatmap for gaze distribution

class Gaze():

    def __init__(self, PATH_GAZE, PATH_BLINK):
        self.PATH_GAZE = PATH_GAZE
        self.PATH_BLINK = PATH_BLINK
        
        # load data
        self.df_gaze = self.load_df(PATH_GAZE)
        self.df_blink = self.load_df(PATH_BLINK)

        # preprocess gaze data
        self.preprocess() 
    
    def load_df(self, fpath):
        return pd.read_csv(fpath)
    
    def get_df(self):
        return self.df_gaze.copy()
    
    def preprocess(self):
        self.reset_using_blink() # reset to sync with vehicle data
        self.filter_norm_pos()   # remove noisy position values
        self.drop_unused()       # drop unused columns

    def reset_using_blink(self):
        # select end frame index from the first instance of a blink longer than 0.5s
        ts_blink = self.df_blink[self.df_blink['duration'] > 0.5].iloc[0]['end_timestamp'] # timestamp when blink ended

        # shift Pupil timestamp to start at t = 0 by removing data before the blink timestamp
        df = self.df_gaze[self.df_gaze['gaze_timestamp'] >= ts_blink].reset_index(drop=True)
        time_bias = df.iloc[0]['gaze_timestamp']
        df['gaze_timestamp'] -= time_bias # to offset the other timestamps
        df = df[df['gaze_timestamp'] > 0.0].reset_index(drop=True) # TODO - figure out non-monotonic data points
        
        self.df_gaze = df
    
    def filter_norm_pos(self):
        ''' Filter out of bound datapoints '''
        self.df_gaze = self.df_gaze[(self.df_gaze['norm_pos_x'] <= 1.0) & (self.df_gaze['norm_pos_x'] >= 0.0)].reset_index(drop=True)
        self.df_gaze = self.df_gaze[(self.df_gaze['norm_pos_y'] <= 1.0) & (self.df_gaze['norm_pos_y'] >= 0.0)].reset_index(drop=True)

    def drop_unused(self):
        ''' Drop columns that are not used '''
        self.df_gaze = self.df_gaze[["gaze_timestamp", "confidence", 
                                     "norm_pos_x", "norm_pos_y", 
                                     "gaze_point_3d_x",	"gaze_point_3d_y", "gaze_point_3d_z"]]
    
    def find_closest_ts_idx(self, ts_veh):
        ''' find closest gaze timestamp w.r.t. vehicle timestamp '''
        return abs(self.df_gaze['gaze_timestamp'] - ts_veh).idxmin()
    
    def vis_norm_pos_x(self):
        plt.figure(figsize=(14, 6))
        # Plot norm_pos_x
        plt.scatter(self.df_gaze['gaze_timestamp'], self.df_gaze['norm_pos_x'])
        plt.xlabel('gaze_timestamp')
        plt.ylabel('norm_pos_x')
        plt.title('Scatter Plot of norm_pos_x')
        plt.show()

    def vis_norm_pos_y(self):
        plt.figure(figsize=(14, 6))
        # Plot norm_pos_y
        plt.scatter(self.df_gaze['gaze_timestamp'], self.df_gaze['norm_pos_y'])
        plt.xlabel('gaze_timestamp')
        plt.ylabel('norm_pos_y')
        plt.title('Scatter Plot of norm_pos_y')
        plt.show()

    def apply_smoothing(self, window_size=11):
        # Add smoothed norm positions columns
        self.df_gaze['smoothed_norm_pos_x'] = self.df_gaze['norm_pos_x'].rolling(window=window_size, min_periods=1, center=True).mean()
        self.df_gaze['smoothed_norm_pos_y'] = self.df_gaze['norm_pos_y'].rolling(window=window_size, min_periods=1, center=True).mean()
    
    def veh_ts_to_gaze_ts(self, veh_ts):
        
        gaze_timestamps = self.df_gaze['gaze_timestamp'].values
        gaze_ts = np.zeros_like(veh_ts)
        gaze_idx = np.zeros_like(veh_ts, dtype=int)

        for i, ts in enumerate(veh_ts):
            closest_idx = (np.abs(gaze_timestamps - ts)).argmin()
            gaze_ts[i] = gaze_timestamps[closest_idx]
            gaze_idx[i] = closest_idx

        return gaze_ts, gaze_idx

    def grab_window(self, gaze_ts, window_size=11):
        center_idx = self.find_closest_ts_idx(gaze_ts)
        start_idx = max(0, center_idx - window_size // 2)
        end_idx = min(len(self.df_gaze) - 1, center_idx + window_size // 2)
        windowed_gaze_data = self.df_gaze.iloc[start_idx:end_idx + 1].copy()
        return windowed_gaze_data

    # TODO - Try better probability distribution that is similar to human gaze distribution    
    def gen_gaze_map(self, gaze_idx, image_size=(1280, 360), WINDOW_SIZE = 5, bandwidth=30, smoothing=1.5, N_LAPS = 10):
        """
        Generate a gaze map at current timestep ts based on smoothed gaze points using KDE.
        - image_size (tuple): Size of the image/frame (width, height) for scaling gaze points
        - bandwidth (int): KDE bandwidth for density estimation (default: 30)
        - smoothing (float): Smoothing factor for the heatmap (default: 1.5)
        """
        print("\nCreating gaze maps ...")
        heatmap_lst = []

        # WINDOW_SIZE points before and after -> Total of 2*WINDOW_SIZE + 1 points considered
        for idx in gaze_idx:
            gaze_points = self.df_gaze.iloc[max(0, idx - WINDOW_SIZE):min(len(self.df_gaze) - 1, idx + WINDOW_SIZE) + 1].copy()
            gaze_points.loc[:, 'scaled_x'] = (gaze_points['norm_pos_x'] * image_size[0]).astype(int)
            gaze_points.loc[:, 'scaled_y'] = (gaze_points['norm_pos_y'] * image_size[1]).astype(int)
            heatmap, xedges, yedges = np.histogram2d(gaze_points['scaled_x'], gaze_points['scaled_y'], bins=[image_size[0]//10, image_size[1]//10])
            heatmap = gaussian_filter(heatmap, sigma=smoothing)
            heatmap_lst.append(heatmap)
        print("Processed gaze maps.")
        return heatmap_lst

    # TODO - Gaze data representation