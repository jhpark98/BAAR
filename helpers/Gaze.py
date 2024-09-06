import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Gaze():

    def __init__(self, PATH_GAZE, PATH_BLINK, PATH_PUPIL): ###
        self.PATH_GAZE = PATH_GAZE
        self.PATH_BLINK = PATH_BLINK
        self.PATH_PUPIL = PATH_PUPIL ###
        
        self.df_gaze = self.load_df(PATH_GAZE)
        self.df_blink = self.load_df(PATH_BLINK)
        self.df_pupil = self.load_df(PATH_PUPIL) ###
        self.preprocess() # preprocess gaze data
    
    def load_df(self, fpath):
        return pd.read_csv(fpath)
    
    def get_df(self):
        return self.df_gaze
    
    def preprocess(self):
        self.reset_using_blink()
        self.filter_norm_pos()
        self.drop_unused()

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
        ''' Filter out of bound datapoints'''
        self.df_gaze = self.df_gaze[(self.df_gaze['norm_pos_x'] <= 1.0) & (self.df_gaze['norm_pos_x'] >= 0.0)].reset_index(drop=True)
        self.df_gaze = self.df_gaze[(self.df_gaze['norm_pos_y'] <= 1.0) & (self.df_gaze['norm_pos_y'] >= 0.0)].reset_index(drop=True)

    def drop_unused(self):
        self.df_gaze = self.df_gaze[["gaze_timestamp", "confidence", 
                                     "norm_pos_x", "norm_pos_y", 
                                     "gaze_point_3d_x",	"gaze_point_3d_y", "gaze_point_3d_z"]]
    
    def find_closest_ts_idx(self, ts_vehicle):
        tmp = abs(self.df_gaze['gaze_timestamp'] - ts_vehicle)
        closest_index = tmp.idxmin()

        return closest_index        
    
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

    def add_ang_vel(self):
        self.df_pupil = self.df_pupil[["pupil_timestamp","eye_id","method","theta","phi"]]
        # Filter by eye 0 and 3D method
        self.df_pupil = self.df_pupil[(self.df_pupil['eye_id'] == 0) & (self.df_pupil['method'] == 'pye3d 0.3.0 real-time')]
        # Calculate angular velocity
        self.df_pupil['omega_theta'] = self.df_pupil['theta'].diff()/self.df_pupil['pupil_timestamp'].diff()
        self.df_pupil['omega_phi'] = self.df_pupil['phi'].diff()/self.df_pupil['pupil_timestamp'].diff()
        self.df_pupil = self.df_pupil.dropna()
        self.df_pupil['omega_magnitude'] = np.sqrt(self.df_pupil['omega_theta']**2 + self.df_pupil['omega_phi']**2)
        
        
    # TODO - Gaze data representation
    
