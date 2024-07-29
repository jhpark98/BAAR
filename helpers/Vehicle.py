import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Vehicle():
    def __init__(self, PATH_VEHICLE, N_LAPS):
        self.df_vehicle = self.load_df(PATH_VEHICLE)
        self.N_LAPS = N_LAPS
        self.veh_lap_end_idx = None
        
        self.preprocess()

    def load_df(self, fpath):
        return pd.read_csv(fpath)
    
    def get_df(self):
        return self.df_vehicle
    
    def preprocess(self):
        self.df_vehicle = self.df_vehicle.iloc[5:].reset_index(drop=True) # remove first row - not necessary datapoint
        self.rename()
        self.reset()
        self.add_brake()
        self.add_distance_traveled()
        

    def rename(self):
        self.df_vehicle = self.df_vehicle.rename(columns={"Pos_x_Vehicle_CoorSys_E_m_": "ego_x", 
                                                          "Pos_y_Vehicle_CoorSys_E_m_": "ego_y",
                                                          "Pos_z_Vehicle_CoorSys_E_m_": "ego_z",
                                                          "a_y_Vehicle_CoG_m_s2_": "a_y_Vehicle_CoG[m|s2]",
                                                          "YawRate_Vehicle_CoG_deg_s_": "YawRate_Vehicle_CoG[deg|s]",
                                                          "v_x_Vehicle_CoG_km_h_": "v_x_Vehicle_CoG[km|h]",
                                                          "s_Vehicle_m_": 's [m]'})
    
    def reset(self):
        # Remove data before starting
        # Use break-off signal to discard section before
        self.df_vehicle = self.df_vehicle[(self.df_vehicle['Pos_BrakePedal___'] == 0.0) & (self.df_vehicle.index > 1000)].reset_index(drop=True) 
        self.df_vehicle.loc[:, 'time_real'] = self.df_vehicle['time_real'] - self.df_vehicle['time_real'].iloc[0]
    
    def add_brake(self):
        self.df_vehicle['Brake[Bar]_avg'] = self.df_vehicle[['p_FL_Brake_bar_', 'p_FR_Brake_bar_', 'p_RL_Brake_bar_', 'p_RR_Brake_bar_']].mean(axis=1)

    def add_distance_traveled(self):
        # Compute the difference in each time step
        self.df_vehicle['ego_x_diff'] = self.df_vehicle['ego_x'].diff()
        self.df_vehicle['ego_y_diff'] = self.df_vehicle['ego_y'].diff()
        self.df_vehicle[['ego_x_diff', 'ego_y_diff']] = self.df_vehicle[['ego_x_diff', 'ego_y_diff']].fillna(0)

        self.df_vehicle['delta_s'] = np.sqrt(self.df_vehicle['ego_x_diff']**2 + self.df_vehicle['ego_y_diff']**2) # distance between two timestep
        self.df_vehicle['s'] = self.df_vehicle['delta_s'].cumsum()

    def find_veh_lap_end_idx(self):
        ''' Find index in df_vehicle where the vehicle complese a single lap'''
        self.df_vehicle.loc[:, 'ego_y_offset'] = self.df_vehicle['ego_y'] - self.df_vehicle['ego_y'].iloc[0]
        sign_changes = np.where((self.df_vehicle['ego_y_offset'].shift(1) < 0) & (self.df_vehicle['ego_y_offset'] >= 0))[0]

        assert self.N_LAPS == len(sign_changes)
        self.veh_lap_end_idx = pd.Index(sign_changes)
        
        return self.veh_lap_end_idx

    def find_gaze_lap_end_idx(self, df_gaze):
        '''Find index of the closes timestep in df_gaze for each index '''
        gaze_lap_end_idx = []
        veh_lap_end_ts = np.array(self.df_vehicle.loc[self.veh_lap_end_idx]["time_real"])
        for ts in veh_lap_end_ts:
            gaze_lap_end_idx.append((np.abs(df_gaze['gaze_timestamp'] - ts)).idxmin())
        
        assert self.N_LAPS == len(gaze_lap_end_idx)

        return pd.Index(gaze_lap_end_idx)

    def vis_map(self):
        # plt.plot(df_vehicle['ego_x'], df_vehicle['ego_y'])
        # plt.plot(df_vehicle['ego_x'][:10000], df_vehicle['ego_y'][:10000], 'dodgerblue', marker='.') 

        start_pos = self.df_vehicle.iloc[0][['ego_x', 'ego_y']]
        start_pos_x, start_pos_y = start_pos['ego_x'], start_pos['ego_y']

        plt.plot([start_pos_x-5.0, start_pos_x+5.0], [start_pos_y, start_pos_y], 'k', marker='o') # starting line
        plt.xlabel('Ego X')
        plt.ylabel('Ego Y')
        plt.title('Ego Position')
        plt.show()