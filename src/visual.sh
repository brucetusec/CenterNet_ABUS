END=277
for i in $(seq 0 $END);
do
    python visualize.py -i $i -s ../visualization/;
done