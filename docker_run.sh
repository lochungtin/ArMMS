# edit these fields to match the directories and files you are using

DATA_FOLDER="data/data_small_groups"
OUTPUT_FOLDER="out"
JOB_FILE="jobs/job_docker.json"

# run docker container

docker run \
    -v $(pwd)/$JOB_FILE:/app/job.json \
    -v $(pwd)/$DATA_FOLDER:/app/data \
    -v $(pwd)/$OUTPUT_FOLDER:/app/out \
    enigmaoffline/armms
