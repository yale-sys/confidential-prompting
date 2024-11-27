import time 
import torch
from collections import defaultdict

log_data = []
cur_time = None

def start_measure():
    global cur_time
    cur_time = time.time()
    

def log_measure(entry):
    global log_data
    global cur_time
    torch.cuda.synchronize()
    elasped = time.time() - cur_time
    
    log_data.append({
        "entry": entry,
        "time": elasped
    })
    cur_time = time.time()
    
def clear():
    global log_data
    log_data = []
    
def read():
    global log_data
    return log_data



def process_data(data):
    # Initialize a defaultdict to store the sum of times for each entry
    time_sums = defaultdict(float)
    
    # Iterate over each dictionary in the input list
    for item in data:
        entry = item['entry']
        time = item['time']
        # Sum the time for each unique entry
        time_sums[entry] += time
    
    # Convert the defaultdict to a regular dictionary for the final output
    return dict(time_sums)