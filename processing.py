import matlab.engine
import argparse

eng = matlab.engine.start_matlab()
parser = argparse.ArgumentParser()

#positional arguments
parser.add_argument("-vehicle_data", help="filename of vehicle data .mat file. For now, place this file in the same directory as processing.py")

#optional arguments
parser.add_argument("-g","--gaze_positions", help="relative filepath to gaze position csv file")
parser.add_argument("-p", "--pupil_positions", help="relative filepath to pupil position csv file")
parser.add_argument("-ei", "--export_info", help="optionally import the eye_tracking export info if needed")

args = parser.parse_args()
eng.extract_data(args.vehicle_data, nargout = 0)

# python3 processing.py -vehicle_data run_4/vehicledynamics.mat