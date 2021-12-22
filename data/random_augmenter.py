import os
import sys
from PIL import Image
from tqdm.auto import tqdm
from transform import augmenter

if __name__ == '__main__':
    sample_dir = sys.argv[1]
    aug_dir = sys.argv[2]
    n_copy = int(sys.argv[3])

    files = sorted(os.listdir(sample_dir))
    for file in tqdm(files):
        name = file.split('.')[0]
        os.makedirs(os.path.join(aug_dir, name), exist_ok=True)
        ori_image = Image.open(os.path.join(sample_dir, file))
        for i in range(n_copy):
            aug_image = augmenter(ori_image)
            aug_image.save(os.path.join(aug_dir, name, f'{name}_{i}.jpg'))
