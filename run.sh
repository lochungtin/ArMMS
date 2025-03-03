# -t aruco marker type
# -s aruco marker size in any unit
# -j image list json file
# -o output directory
# -g generate result images
# -x generate overlaid images
# -a enable advanced thresholding
# -d do detection only
# -v verbose output


python inference.py \
    -r "/Volumes/PhD Data/Reproducibility Study/development_v1/Participant1Anon/" \
    -o out/ \
    -i 0 1 2 3 4 5 6 7\
    -g 1 \
    -x 1 \
    -d 1
