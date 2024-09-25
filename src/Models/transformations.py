import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

def get_transforms(image_size):

    data_transforms = {
        "train": A.Compose([
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=(-180,180), p=0.5),
            A.HueSaturationValue(
                    hue_shift_limit=0.1, 
                    sat_shift_limit=0.1, 
                    val_shift_limit=0.1, 
                    p=0.2
                ),
            A.RandomBrightnessContrast(
                    brightness_limit=(-0.1,0.1), 
                    contrast_limit=(-0.1, 0.1), 
                    p=0.2
                ),
            A.Resize(image_size, image_size),
            A.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225], 
                    max_pixel_value=255.0, 
                    p=1.0
                ),
            ToTensorV2()], p=1.),
        
        "valid": A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225], 
                    max_pixel_value=255.0, 
                    p=1.0
                ),
            ToTensorV2()], p=1.)
    }

    return data_transforms

def get_tta_transforms(image_size):

    tta_transforms = [
        A.Compose([A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
                A.Resize(image_size, image_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225], 
                    max_pixel_value=255.0, 
                    p=1.0),
                ToTensorV2()], p=1.),
        
        A.Compose([A.HorizontalFlip(p=1.0),
                A.Resize(image_size, image_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225], 
                    max_pixel_value=255.0, 
                    p=1.0),
                ToTensorV2()], p=1.),
        
        A.Compose([A.VerticalFlip(p=1.0),
                A.Resize(image_size, image_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225], 
                    max_pixel_value=255.0, 
                    p=1.0),
                ToTensorV2()], p=1.),
        
        A.Compose([A.RandomRotate90(p=1),
                A.Resize(image_size, image_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225], 
                    max_pixel_value=255.0, 
                    p=1.0),
                ToTensorV2()], p=1.),
    ]

    return tta_transforms