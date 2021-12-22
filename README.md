# UDA-Image-classification
A simple pytorch implementation of [UDA](https://arxiv.org/pdf/1904.12848.pdf) for image classification.

# Steps
- Install
> pip install -r requirements.txt

- Generate training data
> python data/random_augmenter.py "sample_directory" "output_augment_directory" "number_of_copies_per_sample"
>
> python data/random_augmenter.py data/images data/aug_images 100

- Training
> Change to your model
> 
> Change config.py (attent at the paths in label_data, unlabel_data)
> 
> run train_uda.py
