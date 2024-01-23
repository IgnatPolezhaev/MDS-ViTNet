<h1>MDS-ViTNet: Improving saliency prediction for Eye-Tracking with Vision Transformer</h1>

In this [paper](link), we present a novel methodology we call MDS-ViTNet (Multi Decoder Saliency by Vision Transformer Network) for enhancing visual saliency prediction or eye-tracking. This approach holds significant potential for diverse fields, including marketing, medicine, robotics, and retail. We propose a network architecture that leverages the Vision Transformer, moving beyond the conventional ImageNet backbone. The framework adopts an encoder-decoder structure, with the encoder utilizing a Swin transformer to efficiently embed most important features. This process involves a Transfer Learning method, wherein layers from the Vision Transformer are converted by the Encoder Transformer and seamlessly integrated into a CNN Decoder. This methodology ensures minimal information loss from the original input image. The decoder employs a multi-decoding technique, utilizing dual decoders to generate two distinct attention maps. These maps are subsequently combined into a singular output via an additional CNN model. Our trained model MDS-ViTNet achieves SoTA results across several benchmarks. Committed to fostering further collaboration, we intend to make our code, models, and datasets accessible to the public.

---

</div>
  
## Results
Example of model work on an image from SALICON.

<table>
<tr>
   <td> 
      <img src="examples/ex_1.png">
   </td>
</tr>
</table>

## MDS-ViTNet architecture
![architecture](examples/MDS-ViTNet.png)

## Dependencies and Installation

1. Clone Repo

   ```bash
   git clone https://github.com/IgnatPolezhaev/MDS-ViTNet.git
   ```

2. Create Conda Environment and Install Dependencies

   ```bash
   # create new anaconda env
   conda create -n mdsvitnet python=3.8 -y
   conda activate mdsvitnet

   # install python dependencies
   pip3 install -r requirements.txt
   ```

## Get Started
### Prepare pretrained models
Download our pretrained models from [Google Drive](link) to the `weights` folder.

The directory structure will be arranged condaas:
```
weights
   |- CNNMerge.pth
   |- ViT_multidecoder.pth
```

### Quick test
We provide some examples in the [`inputs`](./inputs) folder. 
Run the following commands to try it out:
```shell
python inference.py --img_path inputs/img_1.jpg
```
The results will be saved in the `results` folder.

For the best result quality, feed images to the model with an aspect ratio of 3:4.
Inference takes up about 10 Gb of GPU

### Dataset preparation
Download train/val/test data from [Google Drive](link). Unzip downloaded datasets files to `datasets`.

The `datasets` directory structure will be arranged as:
```
datasets
   |- train
      |- train_images
         |- 00000.jpg
         |- 00001.jpg
      |- train_maps
         |- 00000.png
         |- 00001.png   
   |- val
      |- val_images
         |- 00000.jpg
         |- 00001.jpg
      |- val_maps
         |- 00000.png
         |- 00001.png 
   |- test
      |- test_images_salicon
         |- 00000.png
         |- 00001.png 
```

### Training
You can use our notebooks `train_multidecoder_model.ipynb` and `train_add_model.ipynb` to train a custom model.

## Contact
If you have any questions, please feel free to reach me out at `polezhaev.im@phystech.edu`. 

## Acknowledgement
This code is based on [TranSalNet](https://github.com/LJOVO/TranSalNet).
