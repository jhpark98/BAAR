function extract_data(filename)

    df = load(filename);

    % time_sim  = df.timestamps.Time; % simulation time 
    time_real = df.timestamps.Data; % real time 

    df_table = table(time_sim, time_real); % create a table to save as a .csv format

    df_vehicle = df.drivingsimout;

    fns = fieldnames(df_vehicle);

    % Extract raw data from each cell array
    for i=1:1:length(fns)
        fieldName = fns {i};
        fieldData = df_vehicle.(fieldName).Data;
        df_table.(fieldName) = fieldData
    end

    % save .csv file
    fname = 'df_vehicle.csv'
    writetable(df_table, fname);
    disp('Processed and .mat file and saved as .csv file.')

end 