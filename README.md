# Modality Plug-and-Play: Elastic Modality Adaptation in Multimodal LLMs for Embodied AI

## Introduction
This is the official code repository for the paper ["Modality Plug-and-Play: Elastic Modality Adaptation in Multimodal LLMs for Embodied AI"](https://arxiv.org/abs/2312.07886). We present mPnP-LLM to enable fully elastic modality adaptation for LLMs via trainable latent connctions. We evaluate the performance of mPnP-LLM on a mini-split of nuScenes-QA dataset with two sensory modalities: RGB camera views and LiDAR point clouds.

## Requirements
Install pytorch first and then install nuscenes-devkit with
```
pip install nuscenes-devkit
```
Install all requirements with
```
pip install -r requirements.txt
```
There might be some requirements missing in the file. Please refer to the error logs when running our code.

## Creating nuScenes-QA-mini
The dataset we used in our experiments is adapted from the [nuScenes-QA dataset v1.0](https://github.com/qiantianwen/NuScenes-QA). To create the train and validation splits for day and night scenes:
* Download nuScenes-mini split from [nuScenes website](https://www.nuscenes.org/)

* Navigate to `nuqamini` folder and create path `nuqamini/dataset/` and move the extracted nuScenes-mini split to it. The correct path of the dataset should look like `nuqamini/dataset/v1.0-mini/data/sets/nuscenes/`. Then create a path of `nuqamini/dataset/v1.0-mini/data/sets/range_projection_outputs/`.

* Navigate to `nuqamini` folder and run `mini_lidar_dataset_creator.py` to generate range projection of the LiDAR point cloud.

* Navigate to `nuqamini` folder and run `nuqamini_dataset_create.ipynb`. Four data splits will be created in Arrow format in the directories:
    ```
    day/train/
    day/validation/
    night_80dimgaussian7/train/
    night_80dimgaussian7/validation/
    ```

**Alternatively**, you could download our processed dataset from huggingface. Check the dataset page [here](https://huggingface.co/datasets/KevinNotSmile/nuscenes-qa-mini).

## Prepare encoders
We use ViT-small for RGB camera views, which will be automatically downloaded when you run our training code. But we also need a pre-trained [RangeViT](https://github.com/valeoai/rangevit) to perceive the LiDAR inputs. Please download the pre-trained RangeViT [here](https://github.com/valeoai/rangevit/releases/download/v1/model_nuscenes_cs_init.pth) and put the downloaded model file under `model/`.

## Running Modality Adaptation
Navigate to `example/mpnp_llm/`. We first do offline training with **RGB modality** on day-train split and evaluate on both day-validation split and night-validation split:
```
python offline_train.py
```
Due to low accuracy on night-split, we want to switch to **LiDAR modality** for better perception:
```
python switch_lidar.py
```
Alternatively, we can include both **RGB and LiDAR modalities**:
```
python add_lidar.py
```
Since we generate a relatively small dataset for training and validation, the obtained accuracy may have small variations due to randomness.

## Citation
```
@article{huang2023modality,
  title={Modality Plug-and-Play: Elastic Modality Adaptation in Multimodal LLMs for Embodied AI},
  author={Huang, Kai and Yang, Boyuan and Gao, Wei},
  journal={arXiv preprint arXiv:2312.07886},
  year={2023}
}
```