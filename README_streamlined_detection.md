
# ConceptGraphs: Open-Vocabulary 3D Scene Graphs for Perception and Planning

This repository contains the code for the ConceptGraphs project. ConceptGraphs builds open-vocabulary 3D scenegraphs that enable a broad range of perception and task planning capabilities.

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

## Installation

You need to install three different repositories to run the code. This is because conceptgraphs depends on gradslam and chamferdist. We will refer to your chosen location for installing repositories as  `/path/to/code`. You you want to install your three repositories as follows:

```bash
/path/to/code/conceptgraphs/
/path/to/code/gradslam/
/path/to/code/chamferdist/
```


Sometimes certain versions of ubuntu/windows, python, pytorch and cuda may not work well together. Unfortunately this means you may need to do some trial and error to get everything working. We have included the versions of the packages we used on our machines, which ran Ubuntu 20.04.

We also recommend using [conda](https://www.anaconda.com/download) to manage your python environment. It creates a separate environment for each project, which is useful for managing dependencies and ensuring that you don't have conflicting versions of packages from other projects.

Run the following commands:

```bash
# Create the conda environment
conda create -n conceptgraph anaconda python=3.10
conda activate conceptgraph

# Install the required libraries
pip install tyro open_clip_torch wandb h5py openai hydra-core distinctipy ultralytics supervision

# Install the Faiss library (CPU version should be fine), this is used for quick indexing of pointclouds for duplicate object matching and merging
conda install -c pytorch faiss-cpu=1.7.4 mkl=2021 blas=1.0=mkl

##### Install Pytorch according to your own setup #####
# For example, if you have a GPU with CUDA 11.8 (We tested it Pytorch 2.0.1)
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia

# Install Pytorch3D (https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)
# conda install pytorch3d -c pytorch3d # This detects a conflict. You can use the command below, maybe with a different version
conda install https://anaconda.org/pytorch3d/pytorch3d/0.7.4/download/linux-64/pytorch3d-0.7.4-py310_cu118_pyt201.tar.bz2

# Install the gradslam package and its dependencies
# You do not need to install them inside the conceptgraph repository folder
# Treat them as separate packages
# Installing chamferdist, if this fails, its not a huge deal, just move on with the installation
cd /path/to/code/
git clone https://github.com/krrish94/chamferdist.git
cd chamferdist
pip install .

# Installing gradslam, make sure to checkout the conceptfusion branch
cd /path/to/code/
git clone https://github.com/gradslam/gradslam.git
cd gradslam
git checkout conceptfusion
pip install .

# We find that cuda development toolkit is the least problemantic way to install cuda. 
# Make sure the version you install is at least close to your cuda version. 
# See here: https://anaconda.org/conda-forge/cudatoolkit-dev
conda install -c conda-forge cudatoolkit-dev

# You also need to ensure that the installed packages can find the right cuda installation.
# You can do this by setting the CUDA_HOME environment variable.
# You can manually set it to the python environment you are using, or set it to the conda prefix of the environment.
export CUDA_HOME=/path/to/anaconda3/envs/conceptgraph/

# Finally install conceptgraphs
cd /path/to/code/
git clone git@github.com:concept-graphs/concept-graphs.git
cd concept-graphs
pip install -e .
```

Now you will need some data to run the code on, the easiest one to use is the [Replica](https://github.com/facebookresearch/Replica-Dataset). You can install it by using the following commands:

```bash
cd /path/to/data
# you can also download the Replica.zip manually through
# link: https://caiyun.139.com/m/i?1A5Ch5C3abNiL password: v3fY (the zip is split into smaller zips because of the size limitation of caiyun)
wget https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip
unzip Replica.zip
```

And now you will need to update the paths in the configuration files in the `conceptgraph/hydra_configs` directory to point to your paths. Which is discussed below:

## Usage

Conceptgraphs creates the scene graph structure in a few steps. First it **detects** objects in the scene, then the **mapping** process creates a 3D object-based pointcloud map of the scene, and then it adds the **edges** for mapped objects to build the scene graph.

**After you have changed the needed configuration values**, you can run a script with a simple command, for example to run detections:

```bash
python scripts/streamlined_detections.py
```


## Setting up your configuration 
We use the [hydra](https://hydra.cc/) package to manage the configuration, so you don't have to give it a bunch of command line arguments, just edit the  entries in the corresponding `.yaml` file in `./conceptgraph/hydra_configs/` and run the script.

For example here is my `./conceptgraph/hydra_configs/streamlined_detections.yaml` file:

```yaml
defaults:
  - base
  - base_mapping
  - replica
  - sam
  - classes
  - logging_level
  - _self_

detections_exp_suffix: exp_s_detections_stride50_yes_bg_44_mr
stride: 50
exp_suffix: s_mapping_yes_bg_multirun_49
save_video: !!bool True
save_objects_all_frames: !!bool True
# merge_interval: 5
denoise_interval: 5

hydra:
  verbose: true
  mode: MULTIRUN
  sweeper:
    params:
      scene_id: room0, office0 # 
```

First the values are loaded from `base.yaml`, then `base_mapping.yaml` then `replica.yaml` and so on. If there is a conflict (i.e. two files are modifying the same config parameter), the values from the earlier file are overwritten. i.e. `replica.yaml` will overwrite any confliting values in `base.yaml` and so on.

Finally `_self_` is loaded, which are te values in `streamlined_detections.yaml` itself. This is where you can put your own custom values. Also feel free to add your own `.yaml` files to `./conceptgraph/hydra_configs/` and they will be loaded in the same way.

To run the detections script, you need to edit the paths for the replica dataset in the `replica.yaml` file. Here is an example of my `concept-graphs/conceptgraph/hydra_configs/replica.yaml` file, you need to change these paths to point to where you have installed the replica dataset:

```yaml
dataset_root: /home/kuwajerw/new_local_data/new_replica/Replica
dataset_config: /home/kuwajerw/repos/new_conceptgraphs/concept-graphs/conceptgraph/dataset/dataconfigs/replica/replica.yaml
scene_id: room0
render_camera_path: /home/kuwajerw/repos/new_conceptgraphs/concept-graphs/conceptgraph/dataset/dataconfigs/replica/replica_room0.json
```


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
