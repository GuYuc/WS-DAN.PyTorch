# WS-DAN.PyTorch
A PyTorch implementation of WS-DAN (Weakly Supervised Data Augmentation Network) for FGVC (Fine-Grained Visual Classification). (_Hu et al._, ["See Better Before Looking Closer: Weakly Supervised Data Augmentation
Network for Fine-Grained Visual Classification"](https://arxiv.org/abs/1901.09891v2), arXiv:1901.09891)

**NOTICE: This is NOT an official implementation by authors of WS-DAN. The official implementation is available at [tau-yihouxiang/WS_DAN](https://github.com/tau-yihouxiang/WS_DAN).**

**UPDATE (Nov 2019)**: Experiments on Aircraft/Bird/Car/Dog datasets and code reorganization.




## Innovations
1. Data Augmentation: Attention Cropping and Attention Dropping
    <div align="center">
    <img src="./images/Fig1.png" height="500px" alt="Fig1" >
    </div>

2. Bilinear Attention Pooling (BAP) for Features Generation
    <div align="center">
    <img src="./images/Fig3.PNG" height="400px" alt="Fig3" >
    </div>

3. Training Process and Testing Process 
    <div align="center">
    <img src="./images/Fig2a.PNG" height="280px" alt="Fig2a" ><img src="./images/Fig2b.PNG" height="250px" alt="Fig2b" >
    </div>



## Performance

|Dataset|Object|Category|Train|Test|Accuracy (Paper)|Accuracy (PyTorch)|Feature Net|
|-------|------|--------|-----|----|----------------|--------------------|---|
|CUB-200-2011|Bird|200|5,994|5,794|89.4|86.68|inception_mixed_6e|
|FGVC-Aircraft|Aircraft|100|6,667|3,333|93.0|-|inception_mixed_6e|
|Stanford Cars|Car|196|8,144|8,041|94.5|-|inception_mixed_6e|
|Stanford Dogs|Dog|120|12,000|8,580|92.2|89.39|inception_mixed_7c|



## Usage

### WS-DAN
This repo contains WS-DAN with feature extractors including VGG19(```'vgg19', 'vgg19_bn'```), 
ResNet34/50/101/152(```'resnet34', 'resnet50', 'resnet101', 'resnet152'```), 
and Inception_v3(```'inception_mixed_6e', 'inception_mixed_7c'```) in PyTorch form, see ```./models/wsdan.py```. 

```python
net = WSDAN(num_classes=num_classes, M=num_attentions, net='inception_mixed_6e', pretrained=True)
net = WSDAN(num_classes=num_classes, M=num_attentions, net='inception_mixed_7c', pretrained=True)
net = WSDAN(num_classes=num_classes, M=num_attentions, net='vgg19_bn', pretrained=True)
net = WSDAN(num_classes=num_classes, M=num_attentions, net='resnet50', pretrained=True)
```



### Dataset Directory

* [FGVC-Aircraft](http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/) (Aircraft)

  ```
  -/FGVC-Aircraft/data/
                  └─── images
                  		└─── 0034309.jpg
                  		└─── 0034958.jpg
                  		└─── ...
                  └─── variants.txt
                  └─── images_variant_trainval.txt
                  └─── images_variant_test.txt
  ```

* [CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) (Bird)

  ```
  -/CUB-200-2011
          └─── images.txt
          └─── image_class_labels.txt
          └─── train_test_split.txt
          └─── images
                  └─── 001.Black_footed_Albatross
                          └─── Black_Footed_Albatross_0001_796111.jpg
                          └─── ...
                  └─── 002.Laysan_Albatross
                  └─── ...
  ```

* [Stanford Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html) (Car)

  ```
  -/StanfordCars
  		└─── cars_test
  				└─── 00001.jpg
                  └─── 00002.jpg
                  └─── ...
          └─── cars_train
  				└─── 00001.jpg
                  └─── 00002.jpg
                  └─── ...
          └─── devkit
          		└─── cars_train_annos.mat
          └─── cars_test_annos_withlabels.mat
  ```

* [Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/) (Dog)

  ```
  -/StanfordDogs
  		└─── Images
  				└─── n02085620-Chihuahua
                          └─── n02085620_10074.jpg
                          └─── ...
                  └─── n02085782-Japanese_spaniel
                  └─── ...
          └─── train_list.mat
          └─── test_list.mat
  ```



### Run

1. ``` git clone``` this repo.
2. Prepare data and **modify DATAPATH** in ```datasets/<abcd>_dataset.py```.
3. Set configurations in ```config.py``` and set correct dataset in ```train.py```:
    ```python
    train_dataset = BirdDataset(phase='train', resize=config.image_size)
    validate_dataset = DogDataset(phase='val', resize=config.image_size)
    ```
4. ```$ nohup python3 train.py > progress.bar &```
5. ```$ tail -f progress.bar``` for training process (other logs are written in ```<config.save_dir>/train.log```).



### Attention Maps Visualization

Codes in ```eval.py``` helps generate attention maps. (Image, Heat Attention Map, Image x Attention Map)

<div align="center">
<img src="./images/007_raw.jpg" height="180px" alt="Raw" ><img src="./images/007_heat_atten.jpg" height="180px" alt="Heat" ><img src="./images/007_raw_atten.jpg" height="180px" alt="Atten" >
</div>

