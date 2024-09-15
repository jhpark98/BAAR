import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CONDITION = {"stop_1": ([55, 65], [-105, -75]), 
             "sharp_cor_1": ([55, 80], [-130, -115]),
             "round_cor_1": ([140, 210], [-130, -70]), 
             "round_cor_2": ([140, 210], [70, 130]),
             "sharp_cor_2": ([55, 80], [75, 130]),
             "stop_2": ([55, 65], [0, 60])}

class Vehicle():
    def __init__(self, PATH_VEHICLE, N_LAPS):
        self.df_vehicle = self.load_df(PATH_VEHICLE)
        self.N_LAPS = N_LAPS
        self.veh_lap_end_idx = [] # list of the index of the end of each lap
        self.veh_index_lst = []   # list of N x (start, end)
        self.section_idx = {}    # dictionary of section - list of N x (start, end)
        
        self.preprocess()

    def load_df(self, fpath):
        return pd.read_csv(fpath)
    
    def get_df(self):
        return self.df_vehicle
    
    def preprocess(self):
        
        # process dataframe
        self.df_vehicle = self.df_vehicle.iloc[5:].reset_index(drop=True) # remove first row - not necessary datapoint
        self.rename()
        self.add_brake() 
        self.add_distance_traveled()
        self.reset() # remove section before and after experiments
        # debug by visualizing v_x
        # def debug():
        #     subset = self.df_vehicle["v_x_Vehicle_CoG[km|h]"]
        #     plt.plot(np.arange(len(subset)), subset)
        #     plt.plot([0, len(subset)], [0, 0], 'r-')
        #     # plt.ylim((-1, 1))
        #     plt.show()
        # debug()

        # sectioning
        self.find_veh_lap_end_idx()
        self.get_intervals(self.veh_lap_end_idx)
        self.find_section_idx()

        # ensure quality of data
        self.quality_check()

    def quality_check(self):
        if self.df_vehicle['delta_s'].describe()["max"] > 0.01:
            print("[WARNING] There is a jump in the trajectory ...")

    def rename(self):
        '''Rename originally Matlab Simulink variable names'''
        self.df_vehicle = self.df_vehicle.rename(columns={"Pos_x_Vehicle_CoorSys_E_m_": "ego_x", 
                                                          "Pos_y_Vehicle_CoorSys_E_m_": "ego_y",
                                                          "Pos_z_Vehicle_CoorSys_E_m_": "ego_z",
                                                          "a_y_Vehicle_CoG_m_s2_": "a_y_Vehicle_CoG[m|s2]",
                                                          "YawRate_Vehicle_CoG_deg_s_": "YawRate_Vehicle_CoG[deg|s]",
                                                          "v_x_Vehicle_CoG_km_h_": "v_x_Vehicle_CoG[km|h]",
                                                          "s_Vehicle_m_": 's [m]'})
    
    def reset(self):
        self.reset_start()
        self.reset_end()

    def reset_start(self):
        '''Sync starting point with the gaze data. 
           Ques used to sync are: 1) (gaze) open eyes after ~5s & 2) (vehicle) break pedal off at the same time
        '''
        # Remove data before experiment
        # Use break-off signal to discard section
        condition = (self.df_vehicle['Pos_BrakePedal___'] == 0.0) & (self.df_vehicle.index > 1000)
        start = self.df_vehicle.index[condition][0]
        self.df_vehicle = self.df_vehicle.iloc[start:].reset_index(drop=True) 
        self.df_vehicle.loc[:, 'time_real'] = self.df_vehicle['time_real'] - self.df_vehicle['time_real'].iloc[0]
        
    def reset_end(self):
        '''
        Sync ending point with the gaze data. Use the last break as the que to sync.
        '''
        subset = self.df_vehicle['v_x_Vehicle_CoG[km|h]'].iloc[-10000:]
        end_index = subset[subset <= 0].index[0]
        self.df_vehicle = self.df_vehicle.iloc[:end_index] # discard the end
    
    def add_brake(self):
        columns = ['p_FL_Brake_bar_', 'p_FR_Brake_bar_', 'p_RL_Brake_bar_', 'p_RR_Brake_bar_']
        self.df_vehicle['Brake[Bar]_avg'] = self.df_vehicle[columns].mean(axis=1)

    def add_distance_traveled(self):
        # Compute the difference in each time step
        self.df_vehicle['ego_x_diff'] = self.df_vehicle['ego_x'].diff()
        self.df_vehicle['ego_y_diff'] = self.df_vehicle['ego_y'].diff()
        self.df_vehicle[['ego_x_diff', 'ego_y_diff']] = self.df_vehicle[['ego_x_diff', 'ego_y_diff']].fillna(0)

        self.df_vehicle['delta_s'] = np.sqrt(self.df_vehicle['ego_x_diff']**2 + self.df_vehicle['ego_y_diff']**2) # distance between two timestep
        self.df_vehicle['s'] = self.df_vehicle['delta_s'].cumsum()

    def find_veh_lap_end_idx(self):
        ''' Find index in df_vehicle where the vehicle start and end a single lap'''
        
        ego_y_start = self.df_vehicle['ego_y'].iloc[0]
        # find indexes where it cross the starting point
        sign_changes = np.where((self.df_vehicle['ego_y'] < ego_y_start) & (self.df_vehicle['ego_y'].shift(1) > ego_y_start))[0]
        # use indexes near the starting point to find where the vehicle actually stopped
        for i, idx in enumerate(sign_changes):
            s = self.df_vehicle['s'].iloc[idx]
            condition = ((self.df_vehicle['s'] > s-30) & (self.df_vehicle['s'] < s))
            indices = self.df_vehicle.index[condition]
            subset = self.df_vehicle['v_x_Vehicle_CoG[km|h]'].iloc[indices]
            sign_changes[i] = subset[subset < 0.0001].index[0]
        
        # add last index 
        sign_changes = np.append(sign_changes, [len(self.df_vehicle)-1])

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

    def find_section_idx(self):
        """
        Find the indices of the sections in the vehicle data based on predefined conditions.
        """
        for condition in CONDITION.keys():
            section_indices = []
            for i in range(len(self.veh_index_lst)):
                _, section_start_idx, section_end_idx = self.get_section(i, condition)
                section_indices.append((section_start_idx, section_end_idx))
            self.section_idx[condition] = np.array(section_indices)

    def vis_map(self):
        # plt.plot(df_vehicle['ego_x'], df_vehicle['ego_y'])
        # plt.plot(df_vehicle['ego_x'][:10000], df_vehicle['ego_y'][:10000], 'dodgerblue', marker='.') 

        start_pos = self.df_vehicle.iloc[0][['ego_x', 'ego_y']]
        start_pos_x, start_pos_y = start_pos['ego_x'], start_pos['ego_y']

        plt.plot([start_pos_x-5.0, start_pos_x+5.0], [start_pos_y, start_pos_y], 'k') # starting line
        plt.xlabel('Ego X')
        plt.ylabel('Ego Y')
        plt.title('Ego Position')
        plt.show()
    
    def vis_section(self, pos_x, pos_y, i):

        # draw a starting line
        start_pos_x, start_pos_y = 60.0, 13.0
        plt.plot([start_pos_x-5.0, start_pos_x+5.0], [start_pos_y, start_pos_y], 'ro-')

        plt.plot(pos_x, pos_y, 'k', marker='.') # all positions
        plt.xlabel('Ego X')
        plt.ylabel('Ego Y')
        plt.title('Ego Position')
        plt.xlim(40, 200)
        plt.ylim(-140, 140)

        # starting and ending points
        plt.plot(pos_x.iloc[0], pos_y.iloc[0], 'c', marker='x', markersize=12, label='Start') # start
        plt.plot(pos_x.iloc[int(0.25*len(pos_x))], pos_y.iloc[int(0.25*len(pos_x))], 'r', marker='^', markersize=12) # 1/4 line
        plt.plot(pos_x.iloc[int(0.5*len(pos_x))], pos_y.iloc[int(0.5*len(pos_x))], 'g', marker='^', markersize=12) # 1/4 line
        plt.plot(pos_x.iloc[int(0.75*len(pos_x))], pos_y.iloc[int(0.75*len(pos_x))], 'b', marker='^', markersize=12) # 1/4 line
        plt.plot(pos_x.iloc[-1], pos_y.iloc[-1], 'y', marker='x', markersize=12, label='Finish') # finish
        plt.legend()

        plt.savefig(f"stop_2_lap_{i+1}.png")
        plt.clf()

    def find_stop_point(self, df_vehicle, e=0.01):
        """
        This function finds stoppoing point of the vehicle and returns the corresponding norm_idx.
            Stopping point is defined when the vehicle speed is less than 0.1 km/h.

        Returns:
            norm_idx: The result of the operation.
        """
        norm_idx = df_vehicle['v_x_Vehicle_CoG[km|h]'].lt(e).idxmax()
        return norm_idx

    def get_intervals(self, idx):
        idx_lst = []
        for i in range(len(idx)):
            if i == 0:
                idx_lst.append((0, idx[i]))
            else:   
                idx_lst.append((idx[i-1], idx[i]))

        self.veh_index_lst = idx_lst
        return self.veh_index_lst

    def get_section(self, i, section, vis=False):
        '''takes an input section to process which part of the map'''

        # select dataframe for i-th lap
        start_idx, end_idx = self.veh_index_lst[i] # lap interval
        df_vehicle_lap = self.get_df().copy(deep=True)
        df_vehicle_lap = df_vehicle_lap[start_idx:end_idx]

        df_section, section_start_idx, section_end_idx = None, None, None

        if section == "": # if not specified, process the entire map
            df_section, section_start_idx, section_end_idx = df_vehicle_lap, start_idx, end_idx
        else:
            # select section using condition
            x_lower, x_upper, y_lower, y_upper = CONDITION[section][0][0], CONDITION[section][0][1], CONDITION[section][1][0], CONDITION[section][1][1]
            x_condition = (df_vehicle_lap['ego_x']>x_lower) & (df_vehicle_lap['ego_x']<x_upper)
            y_condition = (df_vehicle_lap['ego_y']>y_lower) & (df_vehicle_lap['ego_y']<y_upper)
            condition = x_condition & y_condition
            indices = df_vehicle_lap.index[condition]
            
            assert indices.is_monotonic_increasing

            if section == "stop_2":
                section_start_idx, section_end_idx = indices[np.argmax(np.diff(indices)) + 1], end_idx
            else:            
                section_start_idx, section_end_idx = indices[0], indices[-1]

            # grab section
            df_section = self.get_df().iloc[section_start_idx:section_end_idx]
        
        if vis:
            # get ego positions of section and visually verify
            pos_x, pos_y = df_section['ego_x'], df_section['ego_y']
            self.vis_section(pos_x, pos_y, i)
            print(f"saved figure {i}")

        return df_section, section_start_idx, section_end_idx

    def get_ts(self, section, length):
        """
        Given a certain section, return a timestep that is 20 m away from the end of the section
        """
        section_idx = self.section_idx[section] # [] of N x (start, end)
        ts = np.array([])

        for _, (start, end) in enumerate(section_idx):
            df_section = self.get_df().iloc[start:end]
            target_distance = df_section['s'].iloc[-1] - length # length far way from stop
            closest_idx = (df_section['s'] - target_distance).abs().idxmin() - df_section.index[0] 
            ts = np.append(ts, df_section['time_real'].iloc[closest_idx])
            # print(f"Distance between stop: {df_section['s'].iloc[-1] - df_section['s'].iloc[closest_idx]}")
        return ts