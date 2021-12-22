from torchvision import transforms as T
augmenter = T.Compose([
                  T.GaussianBlur(3),
                  T.RandomAffine(degrees=90, scale=(1.2, 1.2), shear=0.1),
                  T.RandomPerspective(distortion_scale = 0.2),
                  T.RandomVerticalFlip(),
                  T.RandomHorizontalFlip(),
                  T.ColorJitter(brightness=0.5, contrast=0.1, saturation=0.1, hue=0.3),
                  T.RandomEqualize(),
                  T.RandomAdjustSharpness(sharpness_factor=2)
])