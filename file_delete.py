# for managing functions related for file deletion
# files will be deleted at around 00:00 EST

# imports
from datetime import datetime, time
from pytz import timezone
import threading
import os
import shutil

# function 1: delete files if condition is reached
def DelFiles(dirs=['user_csvs', 'user_files', 'user_images']):
    for dir in dirs:
        cwd = os.getcwd()
        content = os.listdir(dir)
        for generic_path in content:
            full_path = os.path.join(cwd, dir, generic_path)
            if os.path.isfile(full_path) and generic_path != '.DS_Store':
                os.remove(full_path)
            elif os.path.isdir(full_path) and generic_path != '.DS_Store':
                shutil.rmtree(full_path)
        

# function 2: monitor time to delete files properly
def MonitorTime(run_var = True):
    while True:
        # get current time in est
        tz = timezone('EST')
        date_time = datetime.now(tz)
        current_time = date_time.time()
        
        # set threshold time (30 min window)
        low_bound = time(0, 0, 0)
        high_bound = time(0, 30, 0)
        
        # check if it's time to reset files
        if current_time >= low_bound and current_time <= high_bound:
            DelFiles()
        
        # wait for 30 min
        threading.Event().wait(30*60)
        

if __name__=='__main__':
    DelFiles()
        
        
    
