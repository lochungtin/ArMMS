# -t aruco marker type
# -s aruco marker size in any unit
# -j image list json file
# -o output directory
# -g generate result images
# -x generate overlaid images
# -a enable advanced thresholding
# -d do detection only


python inference.py \
    -j jobs/ex_inference.json \
    -o "out/"\
    -g 1\
    -x 1
