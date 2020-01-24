#!/bin/bash

#Get all options
while getopts u:f:s:e: option
do
case "${option}"
in
u) USER=${OPTARG};;
f) FILEPATH=${OPTARG};;
s) STARTDATE=${OPTARG};;
e) ENDDATE=${OPTARG};;
esac
done

#scp $USER@helios3.calculquebec.ca:/project/cq-training-1/project1/data/catalog.helios.public.20100101-20160101.pkl $FILEPATH/catalog.helios.public.20100101-20160101.pkl 

# Get list of file names
NAMES="$(ssh $USER@helios3.calculquebec.ca ls /project/cq-training-1/project1/data/hdf5v7_8bit)"
#echo $DIR
# Seperate the file names to create an array of names
IFS=' ' read -ra FILE_NAME_ARRAY <<< "$NAMES"

for file in "${FILE_NAME_ARRAY[@]}"
do
    echo "$file"
    if [[ $IFS =~ $STARTDATE ]];
    then
        echo "$file"
        # scp $USER@helios3.calculquebec.ca:/project/cq-training-1/project1/data/hdf5v7_8bit/$IFS $FILEPATH/hdf5v7_8bit/$IFS
    fi
done