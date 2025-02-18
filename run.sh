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
    -r data/ \
    -o out/ \
    -i 0 1 2 3 4 5 6 7\
    -g 1 \
    -x 1
