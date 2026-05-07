# 102 Category Flower Dataset
<h3>Maria-Elena Nilsback and Andrew Zisserman</h3>



## Overview

We have created a 102 category dataset, consisting of 102 flower categories. The flowers chosen to be flower commonly occuring in the United Kingdom. Each class consists of between 40 and 258 images. The details of the categories and the number of images for each class can be found on this category statistics page.

The images have large scale, pose and light variations. In addition, there are categories that have large variations within the category and several very similar categories. The dataset is visualized using isomap with shape and colour features.



## Directory Structure

```
> dataset
	> train
	> valid
	> test
- cat_to_name.json
- README.md
- sample_submission.csv
```



## Visualization of the dataset

We visualize the categories in the dataset using SIFT features as shape descriptors and HSV as colour descriptor. The images are randomly sampled from the category.

![](https://i.imgur.com/Tl6TKUC.png)



## Publications

Nilsback, M-E. and Zisserman, A.
<a href="https://www.robots.ox.ac.uk/~vgg/publications/2008/Nilsback08/">Automated flower classification over a large number of classes</a>  
Proceedings of the Indian Conference on Computer Vision, Graphics and Image Processing (2008)



## Source

* Original source of this data can be found in <a href="https://www.robots.ox.ac.uk/~vgg/data/flowers/102/"> 102 Category Flower Dataset</a>
* Original readme from author can be found in <a href="https://www.robots.ox.ac.uk/~vgg/data/flowers/102/README.txt">AUTHOR README</a>
* Directory test is added from another kaggle dataset that can be found in <a href="https://www.kaggle.com/c/oxford-102-flower-pytorch/">Oxford 102 Flower Pytorch</a>