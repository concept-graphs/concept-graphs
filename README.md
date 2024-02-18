# ConceptGraphs: Open-Vocabulary 3D Scene Graphs for Perception and Planning

[**Project Page**](https://concept-graphs.github.io/) |
[**Paper**](https://concept-graphs.github.io/assets/pdf/2023-ConceptGraphs.pdf) |
[**ArXiv**](https://arxiv.org/abs/2309.16650) |
[**Video**](https://www.youtube.com/watch?v=mRhNkQwRYnc&feature=youtu.be&ab_channel=AliK)


[Qiao Gu](https://georgegu1997.github.io/)\*,
[Ali Kuwajerwala](https://www.alihkw.com/)\*,
[Sacha Morin](https://sachamorin.github.io/)\*,
[Krishna Murthy Jatavallabhula](https://krrish94.github.io/)\*,
[Bipasha Sen](https://bipashasen.github.io/),
[Aditya Agarwal](https://skymanaditya1.github.io/),
[Corban Rivera](https://www.jhuapl.edu/work/our-organization/research-and-exploratory-development/red-staff-directory/corban-rivera),
[William Paul](https://scholar.google.com/citations?user=92bmh84AAAAJ),
[Kirsty Ellis](https://mila.quebec/en/person/kirsty-ellis/),
[Rama Chellappa](https://engineering.jhu.edu/faculty/rama-chellappa/),
[Chuang Gan](https://people.csail.mit.edu/ganchuang/),
[Celso Miguel de Melo](https://celsodemelo.net/),
[Joshua B. Tenenbaum](http://web.mit.edu/cocosci/josh.html),
[Antonio Torralba](https://groups.csail.mit.edu/vision/torralbalab/),
[Florian Shkurti](http://www.cs.toronto.edu//~florian/),
[Liam Paull](http://liampaull.ca/)

![Splash Figure](./assets/splash-final.png)


## Setup

The env variables needed can be found in `env_vars.bash.template`. When following the setup guide below, you can duplicate that files and change the variables accordingly for easy setup. 

### Install the required libraries

We recommend setting up a virtual environment using virtualenv or conda. Our code has been tested with Python 3.10.12. It may also work with other later versions. We also provide the `environment.yml` file for Conda users. In generaly, directly installing conda env using `.yml` file may cause some unexpected issues, so we recommand setting up the environment by the following instructions and only using the `.yml` file as a reference. 

Sample instructions for `conda` users. 

```bash
conda create -n conceptgraph anaconda python=3.10
conda activate conceptgraph

# Install the required libraries
pip install tyro open_clip_torch wandb h5py openai hydra-core

# Install the Faiss library (CPU version should be fine)
conda install -c pytorch faiss-cpu=1.7.4 mkl=2021 blas=1.0=mkl

##### Install Pytorch according to your own setup #####
# For example, if you have a GPU with CUDA 11.8 (We tested it Pytorch 2.0.1)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install Pytorch3D (https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)
# conda install pytorch3d -c pytorch3d # This detects a conflict. You can use the command below, maybe with a different version
conda install https://anaconda.org/pytorch3d/pytorch3d/0.7.4/download/linux-64/pytorch3d-0.7.4-py310_cu118_pyt201.tar.bz2

# Install the gradslam package and its dependencies
git clone https://github.com/krrish94/chamferdist.git
cd chamferdist
pip install .
cd ..
git clone https://github.com/gradslam/gradslam.git
cd gradslam
git checkout conceptfusion
pip install .
```

### Install [Grounded-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything) package

Follow the instructions on the original [repo](https://github.com/IDEA-Research/Grounded-Segment-Anything#install-without-docker). ConceptGraphs has been tested with the codebase at this [commit](https://github.com/IDEA-Research/Grounded-Segment-Anything/commit/a4d76a2b55e348943cba4cd57d7553c354296223). Grounded-SAM codebase at later commits may require some adaptations. 

First checkout the package by 

```bash
git clone git@github.com:IDEA-Research/Grounded-Segment-Anything.git
```

Then, install the package Following the commands listed in the original GitHub repo. You can skip the `Install osx` step and the "optional dependencies". 

During this process, you will need to set the `CUDA_HOME` to be where the CUDA toolkit is installed. 
The CUDA tookit can be set up system-wide or within a conda environment. We tested it within a conda environment, i.e. installing [cudatoolkit-dev](https://anaconda.org/conda-forge/cudatoolkit-dev) using conda. 

```bash
# i.e. You can install cuda toolkit using conda
conda install -c conda-forge cudatoolkit-dev

# and you need to replace `export CUDA_HOME=/path/to/cuda-11.3/` by 
export CUDA_HOME=/path/to/anaconda3/envs/conceptgraph/
```

You also need to download `ram_swin_large_14m.pth`, `groundingdino_swint_ogc.pth`, `sam_vit_h_4b8939.pth` (and optionally `tag2text_swin_14m.pth` if you want to try Tag2Text) following the instruction [here](https://github.com/IDEA-Research/Grounded-Segment-Anything#label-grounded-sam-with-ram-or-tag2text-for-automatic-labeling). 

After installation, set the path to Grounded-SAM as an environment variable

```bash
export GSA_PATH=/path/to/Grounded-Segment-Anything
```

### (Optional) Set up the EfficientSAM variants

Follow the installation instructions on this [page](https://github.com/IDEA-Research/Grounded-Segment-Anything/tree/main/EfficientSAM). The major steps are:

* Install FastSAM codebase following [here](https://github.com/CASIA-IVA-Lab/FastSAM#installation). You don't have to create a new conda env. Just installing it in the same env as the Grounded-SAM is fine.
* Download FastSAM checkpoints [FastSAM-x.pt](https://github.com/CASIA-IVA-Lab/FastSAM#model-checkpoints) and save it to `Grounded-Segment-Anything/EfficientSAM`. 
* Download MobileSAM checkpoints [mobile_sam.pt](https://github.com/ChaoningZhang/MobileSAM/blob/master/weights/mobile_sam.pt) and save it to `Grounded-Segment-Anything/EfficientSAM`. 
* Download Light HQ-SAM checkpoints [sam_hq_vit_tiny.pth](https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_tiny.pth) and save it to `Grounded-Segment-Anything/EfficientSAM`. 


### Install this repo

```bash
git clone git@github.com:concept-graphs/concept-graphs.git
cd concept-graphs
pip install -e .
```

### Set up LLaVA (used for scene graph generation)

Follow the instructions on the [LLaVA repo](https://github.com/haotian-liu/LLaVA) to set it up. You also need to prepare the LLaVA checkpoints and save them to `$LLAVA_MODEL_PATH`. We have tested with model checkpoint `LLaVA-7B-v0` and [LLaVA code](https://github.com/haotian-liu/LLaVA) at this [commit](https://github.com/haotian-liu/LLaVA/commit/8fc54a09a6be74b2abd913c468fb3d42ae826194). LLaVA codebase at later commits may require some adaptations.

```bash
# Set the env variables as follows (change the paths accordingly)
export LLAVA_PYTHON_PATH=/path/to/llava
export LLAVA_MODEL_PATH=/path/to/LLaVA-7B-v0
```

## Prepare dataset (Replica as an example)

ConceptGraphs takes posed RGB-D images as input. Here we show how to prepare the dataset using [Replica](https://github.com/facebookresearch/Replica-Dataset) as an example. Instead of the original Replica dataset, download the scanned RGB-D trajectories of the Replica dataset provided by [Nice-SLAM](https://github.com/cvg/nice-slam). It contains rendered trajectories using the mesh models provided by the original Replica datasets. 

Download the Replica RGB-D scan dataset using the downloading [script](https://github.com/cvg/nice-slam/blob/master/scripts/download_replica.sh) in [Nice-SLAM](https://github.com/cvg/nice-slam#replica-1) and set `$REPLICA_ROOT` to its saved path.

```bash
export REPLICA_ROOT=/path/to/Replica

export CG_FOLDER=/path/to/concept-graphs/
export REPLICA_CONFIG_PATH=${CG_FOLDER}/conceptgraph/dataset/dataconfigs/replica/replica.yaml
```

ConceptGraphs can also be easily run on other dataset. See `dataset/datasets_common.py` for how to write your own dataloader. 

## Run ConceptGraph

The following commands should be run in the `conceptgraph` folder.

```bash
cd conceptgraph
```

### (Optional) Run regular 3D reconstruction for sanity check

The following command runs a 3D RGB reconstruction ([GradSLAM](https://github.com/gradslam/gradslam)) of a replica scene and also visualize it. This is useful for sanity check. 

* `--visualize` requires it to be run with GUI.

```bash
SCENE_NAME=room0
python scripts/run_slam_rgb.py \
    --dataset_root $REPLICA_ROOT \
    --dataset_config $REPLICA_CONFIG_PATH \
    --scene_id $SCENE_NAME \
    --image_height 480 \
    --image_width 640 \
    --stride 5 \
    --visualize
```

### Extract 2D (Detection) Segmentation and per-resgion features

First, (Detection) Segmentation results and per-region CLIP features are extracted. In the following, we provide two options. 
* The first one (ConceptGraphs) uses SAM in the "segment all" mode and extract class-agnostic masks. 
* The second one (ConceptGraphs-Detect) uses a tagging model and a detection model to extract class-aware bounding boxes first, and then use them as prompts for SAM to segment each object. 

```bash
SCENE_NAME=room0

# The CoceptGraphs (without open-vocab detector)
python scripts/generate_gsa_results.py \
    --dataset_root $REPLICA_ROOT \
    --dataset_config $REPLICA_CONFIG_PATH \
    --scene_id $SCENE_NAME \
    --class_set none \
    --stride 5

# The ConceptGraphs-Detect 
CLASS_SET=ram
python scripts/generate_gsa_results.py \
    --dataset_root $REPLICA_ROOT \
    --dataset_config $REPLICA_CONFIG_PATH \
    --scene_id $SCENE_NAME \
    --class_set $CLASS_SET \
    --box_threshold 0.2 \
    --text_threshold 0.2 \
    --stride 5 \
    --add_bg_classes \
    --accumu_classes \
    --exp_suffix withbg_allclasses
```

The above commands will save the detection and segmentation results in `$REPLICA_ROOT/$SCENE_NAME/`. 
The visualization of the detection and segmentation can be viewed in `$REPLICA_ROOT/$SCENE_NAME/gsa_vis_none` and `$REPLICA_ROOT/$SCENE_NAME/gsa_vis_ram_withbg_allclasses` respectively. 

You can ignore the `There's a wrong phrase happen, this is because of our post-process merged wrong tokens, which will be modified in the future. We will assign it with a random label at this time.` message for now. 

### Run the 3D object mapping system

The following command builds an object-based 3D map of the scene, using the image segmentation results from above.  

* Use `save_objects_all_frames=True` to save the mapping results at every frame, which can be used for animated visualization by `scripts/animate_mapping_interactive.py` and `scripts/animate_mapping_save.py`. 
* Use `merge_interval=20  merge_visual_sim_thresh=0.8  merge_text_sim_thresh=0.8` to also perform overlap-based merging during the mapping process. 

```bash
# Using the CoceptGraphs (without open-vocab detector)
THRESHOLD=1.2
python slam/cfslam_pipeline_batch.py \
    dataset_root=$REPLICA_ROOT \
    dataset_config=$REPLICA_CONFIG_PATH \
    stride=5 \
    scene_id=$SCENE_NAME \
    spatial_sim_type=overlap \
    mask_conf_threshold=0.95 \
    match_method=sim_sum \
    sim_threshold=${THRESHOLD} \
    dbscan_eps=0.1 \
    gsa_variant=none \
    class_agnostic=True \
    skip_bg=True \
    max_bbox_area_ratio=0.5 \
    save_suffix=overlap_maskconf0.95_simsum${THRESHOLD}_dbscan.1_merge20_masksub \
    merge_interval=20 \
    merge_visual_sim_thresh=0.8 \
    merge_text_sim_thresh=0.8

# On the ConceptGraphs-Detect 
SCENE_NAMES=room0
THRESHOLD=1.2
python slam/cfslam_pipeline_batch.py \
    dataset_root=$REPLICA_ROOT \
    dataset_config=$REPLICA_CONFIG_PATH \
    stride=5 \
    scene_id=$SCENE_NAME \
    spatial_sim_type=overlap \
    mask_conf_threshold=0.25 \
    match_method=sim_sum \
    sim_threshold=${THRESHOLD} \
    dbscan_eps=0.1 \
    gsa_variant=ram_withbg_allclasses \
    skip_bg=False \
    max_bbox_area_ratio=0.5 \
    save_suffix=overlap_maskconf0.25_simsum${THRESHOLD}_dbscan.1
```

The above commands will save the mapping results in `$REPLICA_ROOT/$SCENE_NAME/pcd_saves`. It will create two `pkl.gz` files, where the one with `_post` suffix indicates results after some post processing, which we recommend using.`

If you run the above command with `save_objects_all_frames=True`, it will create a folder in `$REPLICA_ROOT/$SCENE_NAME/objects_all_frames`. Then you can run the following command to visualize the mapping process or save it to a video. Also see the relevant files for available key callbacks for viusalization options. 

```
python scripts/animate_mapping_interactive.py --input_folder $REPLICA_ROOT/$SCENE_NAME/objects_all_frames/<folder_name>
python scripts/animate_mapping_save.py --input_folder $REPLICA_ROOT/$SCENE_NAME/objects_all_frames/<folder_name>
```

### Visualize the object-based mapping results

```bash
python scripts/visualize_cfslam_results.py --result_path /path/to/output.pkl.gz
```

Then in the open3d visualizer window, you can use the following key callbacks to change the visualization. 
* Press `b` to toggle the background point clouds (wall, floor, ceiling, etc.). Only works on the ConceptGraphs-Detect.
* Press `c` to color the point clouds by the object class from the tagging model. Only works on the ConceptGraphs-Detect.
* Press `r` to color the point clouds by RGB. 
* Press `f` and type text in the terminal, and the point cloud will be colored by the CLIP similarity with the input text. 
* Press `i` to color the point clouds by object instance ID. 

### Evaluate semantic segmentation from the object-based mapping results on Replica datasets

First, download the GT point cloud with per-point semantic segmentation labels from this [Google Drive link](https://drive.google.com/file/d/1NhQIM5PCH5L5vkZDSRq6YF1bRaSX2aem/view?usp=sharing). Please refer to [this issue](https://github.com/concept-graphs/concept-graphs/issues/18#issuecomment-1876673985) for a brief description of how they are generated. Unzip the file and record its location in `REPLICA_SEMANTIC_ROOT`. 

Then run the following command to evaluate the semantic segmentation results. The results will be saved in the `results` folder, where the mean recall `mrecall` is the mAcc and `fmiou` is the F-mIoU reported in the paper. 

```bash
# CoceptGraphs (without open-vocab detector)
python scripts/eval_replica_semseg.py \
    --replica_root $REPLICA_ROOT \
    --replica_semantic_root $REPLICA_SEMANTIC_ROOT \
    --n_exclude 6 \
    --pred_exp_name none_overlap_maskconf0.95_simsum1.2_dbscan.1_merge20_masksub

# On the ConceptGraphs-Detect (Grounding-DINO as the object detector)
python scripts/eval_replica_semseg.py \
    --replica_root $REPLICA_ROOT \
    --replica_semantic_root $REPLICA_SEMANTIC_ROOT \
    --n_exclude 6 \
    --pred_exp_name ram_withbg_allclasses_overlap_maskconf0.25_simsum1.2_dbscan.1_masksub
```



### Extract object captions and build scene graphs

Ensure that the `openai` package is installed and that your APIKEY is set. We recommend using GPT-4, since GPT-3.5 often produces inconsistent results on this task.
```bash
export OPENAI_API_KEY=<your GPT-4 API KEY here>
```

Also note that you may need to make the following change at [this line](https://github.com/haotian-liu/LLaVA/blob/main/llava/mm_utils.py#L68) in the original LLaVa repo to run the following commands. 

```python
            # if output_ids[0, -keyword_id.shape[0]:] == keyword_id:
            #     return True
            if torch.equal(output_ids[0, -keyword_id.shape[0]:], keyword_id):
                return True
```

Then run the following commands sequentially to extract per-object captions and build the 3D scene graph. 

```bash
SCENE_NAME=room0
PKL_FILENAME=output.pkl.gz  # Change this to the actual output file name of the pkl.gz file

python scenegraph/build_scenegraph_cfslam.py \
    --mode extract-node-captions \
    --cachedir ${REPLICA_ROOT}/${SCENE_NAME}/sg_cache \
    --mapfile ${REPLICA_ROOT}/${SCENE_NAME}/pcd_saves/${PKL_FILENAME} \
    --class_names_file ${REPLICA_ROOT}/${SCENE_NAME}/gsa_classes_ram_withbg_allclasses.json

python scenegraph/build_scenegraph_cfslam.py \
    --mode refine-node-captions \
    --cachedir ${REPLICA_ROOT}/${SCENE_NAME}/sg_cache \
    --mapfile ${REPLICA_ROOT}/${SCENE_NAME}/pcd_saves/${PKL_FILENAME} \
    --class_names_file ${REPLICA_ROOT}/${SCENE_NAME}/gsa_classes_ram_withbg_allclasses.json

python scenegraph/build_scenegraph_cfslam.py \
    --mode build-scenegraph \
    --cachedir ${REPLICA_ROOT}/${SCENE_NAME}/sg_cache \
    --mapfile ${REPLICA_ROOT}/${SCENE_NAME}/pcd_saves/${PKL_FILENAME} \
    --class_names_file ${REPLICA_ROOT}/${SCENE_NAME}/gsa_classes_ram_withbg_allclasses.json
```

Then the object map with scene graph can be visualized using the following command. 
* Press `g` to show the scene graph. 
* Press "+" and "-" to increase and decrease the size of point cloud for better visualization.

```bash
python scripts/visualize_cfslam_results.py \
    --result_path ${REPLICA_ROOT}/${SCENE_NAME}/sg_cache/map/scene_map_cfslam_pruned.pkl.gz \
    --edge_file ${REPLICA_ROOT}/${SCENE_NAME}/sg_cache/cfslam_object_relations.json
```


## AI2Thor-related experiments

During the development stage, we performed some experiments on the AI2Thor dataset. 
Upon request, now we provide the code and instructions for these experiments. 
However, note that we didn't perform any quantitative evaluation on AI2Thor. 
And because of domain gap, performance of ConceptGraphs may be worse than other datasets reported. 

### Setup 

Use our own [fork](https://github.com/georgegu1997/ai2thor), where some changes were made to record the interaction trajectories. 

```bash
cd .. # go back to the root folder CFSLAM
git clone git@github.com:georgegu1997/ai2thor.git
cd ai2thor
git checkout main5.0.0
pip install -e .

# This is for the ProcThor dataset.
pip install ai2thor-colab prior --upgrade
```

If you meet error saying `Could not load the Qt platform plugin "xcb"` later on, it probably means that is some weird issue with `opencv-python` and `opencv-python-headless`. Try uninstalling them and install one of them back. 

### Generating AI2Thor datasets

1. Use `$AI2THOR_DATASET_ROOT` as the directory ai2thor dataset and save it to a variable. Also set the scene used from AI2Thor. 

```bash
# Change this to run it in a different scene in AI2Thor environment
# train_3 is a scene from the ProcThor dataset, which containing multiple rooms in one house
SCENE_NAME=train_3

# The following scripts need to be run in the conceptgraph folder
cd ./conceptgraph
```

2. Generate a densely captured grid map for the selected scene. 
```bash
# Uniform sample camera locations (XY + Yaw)
python scripts/generate_ai2thor_dataset.py --dataset_root $AI2THOR_DATASET_ROOT --scene_name $SCENE_NAME --sample_method uniform --n_sample -1 --grid_size 0.5
# Uniform sample + randomize lighting
python scripts/generate_ai2thor_dataset.py --dataset_root $AI2THOR_DATASET_ROOT --scene_name $SCENE_NAME --sample_method uniform --n_sample -1 --grid_size 0.5 --save_suffix randlight --randomize_lighting
```

3. Generate a human-controlled trajectory for the selected scene. (GUI and keyboard interaction needed)
```bash
# Interact generation and save trajectory files. 
# This line will open up a Unity window. You can control the agent with arrow keys in the terminal window. 
python scripts/generate_ai2thor_dataset.py --dataset_root $AI2THOR_DATASET_ROOT --scene_name $SCENE_NAME --interact

# Generate observations from the saved trajectory file
python scripts/generate_ai2thor_dataset.py --dataset_root $AI2THOR_DATASET_ROOT --scene_name $SCENE_NAME --sample_method from_file
```

4. Generate a trajectory with object randomly moved. 
```bash
MOVE_RATIO=0.25
RAND_SUFFIX=mv${MOVE_RATIO}
python scripts/generate_ai2thor_dataset.py \
    --dataset_root $AI2THOR_DATASET_ROOT \
    --scene_name $SCENE_NAME \
    --interact \
    --save_suffix $RAND_SUFFIX \
    --randomize_move_moveable_ratio $MOVE_RATIO \
    --randomize_move_pickupable_ratio $MOVE_RATIO
```