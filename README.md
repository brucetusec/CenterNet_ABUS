# CenterNet adaption for ABUS
This project is the implementation of Tien-Yi Chi's master thesis "A Center Key-point Deep Convolutional Neural Network
of Tumor Detection for Automated Breast Ultrasound Image".
### Directories
+ data
  Contains numpy images and ground truth bounding box info. Please download "sys_ucc" folder from test_data in CAD and place it here.
+ history
  Stores results from the experiments in the past.
+ results
  Stores results produced by inference.py of the current version.
+ src
  Source code.
+ visualization
  Stores images in 3 different views with boxes drawn on them.

---

### In the folder src:
All python operations were executed here; for example, `python inference.py -f 0 -e 10`. The most important ones are trainer.py, inference.py, draw_froc.py, draw_map_by_dist.py, and visualize.py.
+ data
  Contains dataset class.
+ models
  Contains network and loss classes.
+ utils
  Contains utility functions used by other classes.
+ checkpoints
  Stores checkpoints for model weights in intermediate epochs.

---

### Model training
```
python trainer.py --crx_valid 4 --batch_size 4 --max_epoch 20 --lr 0.001 --freeze --resume --resume_ep 10
```
Argument "crx_valid" is the target fold for testing, and training was done on the others folds combined. Other args should be self-explanatory.
### Inference
```
python inference.py -f 1 -e 10
```
Run inference on fold 1 using checkpoint "hourglass_10". Argument f means fold number, and e means target epoch.
### Draw graphs
```
python draw_map_by_dist.py --threshold 20
```
Argument threshold is for size filtering. Default value is 0, while it was set to 20 in the thesis.
### Visualization
```
python visualize.py -i 0 -s ../visualization/
```
This operation draws the 3-D volume in the form of 2-D slices from 3 different views.
The argument i is the index in "data/sys_ucc/annotations/rand_all.txt" of the image being drawn, and s is the fold for saving the drawn images.
### Tests
+ test_model.py
  Prints out model structure and calculates raw memory used (not the actual memory cost of training).  
+ test_flow.py
  Tests if model training is runnable with specified batch size.  
+ test_data.py
  Draws visualized ground truths on image slices.  
The rest aren't necessary, dive further only when you're interested.

---

Please checkout https://github.com/TienYiChi/CenterNet_ABUS for older versions.

Original CenterNet: https://github.com/xingyizhou/CenterNet