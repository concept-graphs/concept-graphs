# Streamlined Detection Script for ConceptGraphs

The `./scripts/streamlined_detections.py` script is an updated version of the `./scripts/generate_gsa_results.py` script. It is both much simpler and runs faster.

## Setting up your configuration 
It uses the `hydra` package to manage the configuration, so you don't have to give it a bunch of command line arguments, just edit the  entries in the corresponding `.yaml` file in `./conceptgraph/hydra_configs/` and run the script.

For example here is my `./conceptgraph/hydra_configs/streamlined_detections.yaml` file:

```yaml
defaults:
  - base
  - replica
  - sam
  - _self_

stride: 1
exp_suffix: _streamlined_yolo_stride50_no_bg24
save_video: true
```

First the values are loaded from `base.yaml`, then `replica.yaml` and so on. If there is a conflict (i.e. two files are modifying the same config parameter), the values from the earlier file are overwritten. i.e. `replica.yaml` will overwrite any confliting values in `base.yaml` and so on.

Finally `_self_` is loaded, which are te values in `streamlined_detections.yaml` itself. This is where you can put your own custom values. Also feel free to add your own `.yaml` files to `./conceptgraph/hydra_configs/` and they will be loaded in the same way.

**IMPORTANT:** Make sure to edit the path values in `replica.yaml` and to point to your own dataset. 

## Running the script

To run the script, simply run the following command from the `conceptgraph` directory:

```bash
python scripts/streamlined_detections.py
```

So for me it looks like this. Note that if you don't have the models installed, it should just automatically download them for you.

```bash
(cg) (base) kuwajerw@kuwajerw-ub20 [09:56:10PM 26/02/2024]:
(main) ~/repos/new_conceptgraphs/concept-graphs/conceptgraph/
$ python scripts/streamlined_detections.py 
Done! Execution time of get_dataset function: 0.09 seconds
Downloading https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l-world.pt to 'yolov8l-world.pt'...
100%|████████████████████| 91.2M/91.2M [00:00<00:00, 117MB/s]
Done! Execution time of YOLO function: 1.32 seconds
Downloading https://github.com/ultralytics/assets/releases/download/v8.1.0/mobile_sam.pt to 'mobile_sam.pt'...
100%|████████████████████| 38.8M/38.8M [00:00<00:00, 115MB/s]
  9%|███████████▋        | 922/2000 [02:13<02:45,  6.50it/s]
```

It will also save a copy of the configuration file in the experiment output directory, so you can see what settings were used for each run. Hope that helps!
