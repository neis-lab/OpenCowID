### OpenCowID: Zero-Shot Visual Identification of Dairy Cows via Metric Learning

#### Setup
pip install -r requirements.txt

#### Stochastic texture generation
To generate synthetic training data using stochastic texture synthesis, run the following command:
```
python generate_data.py --directory_path /path/to/save/images
```

#### Training
Training data folder should contain separate subfolders for each class.
```
python train.py --train_data_folder <path to train data directory> --val_gallery_folder <path to the directory containing reference (gallery) images of IDs in validation set> --val_data_folder <path to directory containing validation set>
```

#### Testing
Construct a gallery directory consisting of reference images (val_and_gallery/gallery/ in the example below). Create a directory with test images (data/val_and_gallery/val/ in the example below)

```
python infer.py --model_path saved_models/synthetic_loss_epoch_0.pth  --train_data_folder data/val_and_gallery/gallery/ --test_data_folder data/val_and_gallery/val/
```

-------

# 🐄 OpenCowID: Zero-Shot Visual Identification of Dairy Cows via Metric Learning

OpenCowID is a lightweight metric-learning framework for **visual identification of dairy cows**, designed to perform well even in **zero-shot** conditions. It uses **stochastic texture synthesis** to create diverse synthetic coat patterns, enabling robust embeddings without requiring large labeled datasets.

---

## 📦 Installation

```
pip install -r requirements.txt
```
## 🎨 Synthetic Coat Pattern Generation
To generate synthetic training data via stochastic texture synthesis, run:
```
python generate_data.py --directory_path /path/to/save/images
```
This script produces randomized cow-coat textures which serve as robust training signals for metric learning.

## Data directory structure

```
data/
└── train_data/
    └── cow_001/
        ├── img_001.jpg
        ├── img_002.jpg
        └── ...
    └── cow_002/
        ├── img_003.jpg
        ├── img_004.jpg
        └── ...
    └── cow_003/
        ├── img_005.jpg
        └── ...
    ...
└── val_and_gallery/
    └──  gallery/               # Reference images per cow ID
        ├── cow_004/
        │   ├── img_001.jpg
        │   └── img_002.jpg
        ├── cow_002/
        │   └── img_001.jpg
        └── ...
    └── val/                   # Query images for validation
        ├── cow_005/
        │   ├── img_001.jpg
        │   └── ...
        ├── cow_006/
        │   └── img_001.jpg
        │   └── ...
        └── ...

└── test_and_gallery/
    └──  gallery/               # Reference images per cow ID
        ├── cow_004/
        │   ├── img_001.jpg
        │   └── img_002.jpg
        ├── cow_002/
        │   └── img_001.jpg
        └── ...
    └── test/                   # Query images for testing
        ├── cow_007/
        │   ├── img_001.jpg
        │   └── ...
        ├── cow_008/
        │   └── img_001.jpg
        │   └── ...
        └── ...
```

## 🏋️‍♂️ Training
The training directory should follow a class-per-folder structure:


Run training with:
```
python train.py \
    --train_data_folder data/train_data \
    --val_gallery_folder data/val_and_gallery/gallery \
    --val_data_folder data/val_and_gallery/test
```


## 🔍 Testing
Prepare two directories:
- A gallery directory with one or more reference images per cow ID
- A test directory with query images.



Testing:
```
python infer.py \
    --model_path <path to model weights> \
    --train_data_folder data/test_and_gallery/gallery/ \
    --test_data_folder data/test_and_gallery/test/

```