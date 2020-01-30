######################################
## Python file is to help download files from helios3 one after an other
## Make sure you can run ssh on your system, I still need to test this on windows.
## Line execution example: >>> python3 downloadData.py -u USER_NAME -f ~/data -s 2011.12
##                This command return the pickel file and all the images from the moth of Feb 2011
##                If you run the script twice, it will not download files again if they don't need to be updated.
##
##
## For windows execution: use ubuntu subsystem for linux and the directory should start with /mnt/c/...
##                        
######################################

import subprocess
import sys, os
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument("-u", help="User Name for ssh")
parser.add_argument("-f", help="File location for loca storage")
parser.add_argument("-s", help="The iamge file name start. Example to get images from the month of Feb in 2011: 2011.02")

args = parser.parse_args()

if not os.path.exists(args.f):
    os.makedirs(args.f)

# Get pickel file
subprocess.run(["rsync", "-P", "-u", args.u + "@helios3.calculquebec.ca:/project/cq-training-1/project1/data/catalog.helios.public.20100101-20160101.pkl", args.f + "/catalog.helios.public.20100101-20160101.pkl"])

# Get files pickture names from remote
fileNames = subprocess.run(["ssh", args.u + "@helios3.calculquebec.ca", "ls", "/project/cq-training-1/project1/data/hdf5v7_8bit"], stdout=subprocess.PIPE)
fileNames = fileNames.stdout.decode().split("\n")


if not os.path.exists(args.f + "/hdf5v7_8bit"):
    os.makedirs(args.f + "/hdf5v7_8bit")

# Get only the ones that match the name that start with the argument passed to the sript
for name in fileNames:
    if name.startswith(args.s):
        print("Getting: " + name)
        subprocess.run(["rsync", "-P", "-u", args.u + "@helios3.calculquebec.ca:/project/cq-training-1/project1/data/hdf5v7_8bit/" + name, args.f + "/hdf5v7_8bit/" + name])
#print (fileNames)


## Update pickel to reflect current directory folders
dataFrame = pd.read_pickle(args.f + "/catalog.helios.public.20100101-20160101.pkl")
print("Updating dataframe... this may take a while!")
for i, row in dataFrame.iterrows():

    if (row.hdf5_8bit_path == "nan"):
        continue
    file_path_names = row.hdf5_8bit_path.split("/")
    new_path = args.f + "/" + "/".join(file_path_names[-2:])
    dataFrame.loc[i, 'hdf5_8bit_path'] = new_path
    
dataFrame.to_pickle(args.f + "/catalog.helios.public.20100101-20160101_updated.pkl")