import pickle
import numpy as np
import os
import torchvision.transforms as transforms
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from datasets.img_aug import RandomRotationWithFlipNP, RandomShearWithFlipNP, RandomSkewWithFlipNP

imgs = {}

class AnimalDataset(Dataset):
    """
    Animals with attributes dataset
    """

    def __init__(
        self,
        classes_file,
        data_path,
        img_dir_list=None,
        transform=None,
        partial_predicates: bool = False,
        num_predicates: int = 85,
        preload: bool = True,
        seed: int = 42,
    ):
        """
        Initializes the dataset object

        @param classes_file: the file listing all classes from the AwA dataset
        @param img_dir_list: list with the file names of images to be included (if set to None all images are included)
        @param transform: transformation applied to the images when loading
        @param partial_predicates: flag identifying whether only a subset of the ground-truth concepts will be observable
        @param num_predicates: if @partial_predicates is True, specifies the number of concepts to be observed
        @param preload:  flag identifying if the images should be preloaded into the CPU memory
        @param seed: random generator seed
        """
        data_path = os.path.join(data_path, "Animals_with_Attributes2")
        predicate_binary_mat = np.array(
            np.genfromtxt(
                os.path.join(data_path, "predicate-matrix-binary.txt"), dtype="int"
            )
        )
        self.predicate_binary_mat = predicate_binary_mat
        self.transform = transform

        # Shall a partial predicate set be used?
        if not partial_predicates:
            self.predicate_idx = np.arange(0, self.predicate_binary_mat.shape[1])
        else:
            np.random.seed(seed)
            self.predicate_idx = np.random.choice(
                a=np.arange(0, self.predicate_binary_mat.shape[1]),
                size=(num_predicates,),
                replace=False,
            )

        class_to_index = dict()
        # Build dictionary of indices to classes
        with open(os.path.join(data_path, "classes.txt")) as f:
            index = 0
            for line in f:
                class_name = line.split("\t")[1].strip()
                class_to_index[class_name] = index
                index += 1
        self.class_to_index = class_to_index

        img_names = []
        img_index = []
        with open(os.path.join(data_path, classes_file)) as f:
            for line in f:
                class_name = line.strip().split()[1]
                FOLDER_DIR = os.path.join(
                    os.path.join(data_path, "JPEGImages"), class_name
                )
                file_descriptor = os.path.join(FOLDER_DIR, "*.jpg")
                files = glob(file_descriptor)

                class_index = class_to_index[class_name]
                for file_name in files:
                    img_names.append(file_name)
                    img_index.append(class_index)

        # If a list of images is pre-specified, use only them
        if img_dir_list is not None:
            inds = [img_names.index(x) for x in img_dir_list if x in img_names]
        else:
            inds = [_ for _ in range(len(img_names))]
        self.img_names = [img_names[i] for i in inds]
        self.img_index = [img_index[i] for i in inds]

        self.preload = preload

        # Preload images if necessary
        if preload:
            print("Pre-loading AwA images...")
            bar = tqdm(total=len(img_names))

            for i in range(len(img_names)):
                if img_names[i] in imgs:
                    pass
                else:
                    im = Image.open(self.img_names[i])
                    if im.getbands()[0] == "L":
                        im = im.convert("RGB")
                    im = im.resize((64, 64))
                    imgs[img_names[i]] = np.array(im)
                bar.update(1)
            bar.close()

    def __getitem__(self, index):
        """
        Returns points from the dataset

        @param index: index
        @return: a dictionary with the data; dict['img_code'] contains indices, dict['file_names'] contains
        image file names, dict['images'] contains images, dict['label'] contains target labels,
        dict['features'] contains images, dict['concepts'] contains concept values.
        """
        if not self.preload:
            im = Image.open(self.img_names[index])
            if im.getbands()[0] == "L":
                im = im.convert("RGB")
        else:
            im = imgs[self.img_names[index]]

        if self.transform:
            im = self.transform(im)

        im_index = self.img_index[index]
        im_predicate = self.predicate_binary_mat[im_index, self.predicate_idx]

        dictionary = {
            "img_code": index,
            "file_names": self.img_names[index],
            "images": im,
            "labels": im_index,
            "features": im,
            "concepts": im_predicate,
        }

        # Return a list of the image and the attribute vector
        return [dictionary['images'], dictionary['concepts']]

    def __len__(self):
        return len(self.img_names)

def train_test_split_AwA(
    classes_file,
    data_path,
    train_ratio=0.6,
    val_ratio=0.2,
    seed=42,
    partial_predicates: bool = False,
    num_predicates: int = 85,
    preload: bool = True,
):
    """
    Performs train-validation-test split and constructs dataset objects

    @param classes_file: the file listing all classes from the AwA dataset
    @param train_ratio: the ratio specifying the training set size in the train-validation-test split
    @param val_ratio: the ratio specifying the validation set size in the train-validation-test split
    @param seed: random generator seed
    @param partial_predicates: flag identifying whether only a subset of the ground-truth concepts will be observable
    @param num_predicates: if @partial_predicates is True, specifies the number of concepts to be observed
    @param preload: flag identifying if the images should be preloaded into the CPU memory
    @return: dataset objects corresponding to the training, validation and test sets, respectively
    """
    assert train_ratio + val_ratio < 1.0
    np.random.seed(seed)
    awa_complete = AnimalDataset(
        classes_file=classes_file,
        data_path=data_path,
        transform=None,
        partial_predicates=partial_predicates,
        num_predicates=num_predicates,
        preload=preload,
        seed=seed,
    )

    img_names_train, img_names_valtest = train_test_split(
        awa_complete.img_names, train_size=train_ratio, random_state=seed
    )
    img_names_val, img_names_test = train_test_split(
        img_names_valtest,
        train_size=val_ratio / (1.0 - train_ratio),
        random_state=2 * seed,
    )

    return img_names_train, img_names_val, img_names_test 

def get_AwA_dataloaders(
    classes_file,
    data_path,
    batch_size,
    train_ratio=0.6,
    val_ratio=0.2,
    seed=42,
    img_size: int = 224,
    partial_predicates: bool = False,
    num_predicates: int = 85,
    preload: bool = True,
    img_aug: bool = True,
):
    """
    Constructs data loaders for the AwA dataset

    @param classes_file: the file listing all classes from the AwA dataset
    @param batch_size: batch size
    @param num_workers: number of worker processes
    @param train_ratio: the ratio specifying the training set size in the train-validation-test split
    @param val_ratio: the ratio specifying the validation set size in the train-validation-test split
    @param seed: random generator seed
    @param partial_predicates: flag identifying whether only a subset of the ground-truth concepts will be observable
    @param num_predicates: if @partial_predicates is True, specifies the number of concepts to be observed
    @param preload: flag identifying if the images should be preloaded into the CPU memory
    @return: a dictionary with the data loaders for the training, validation and test sets
    """
    # Train-validation-test split
    img_names_train, img_names_val, img_names_test = train_test_split_AwA(
        classes_file=classes_file,
        data_path=data_path,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        partial_predicates=partial_predicates,
        num_predicates=num_predicates,
        preload=preload,
    )

    # Transformations
    transform_list_train = []
    if not preload:
        transform_list_train.append(transforms.Resize(size=(img_size, img_size)))
    transform_list_train.append(RandomRotationWithFlipNP(degrees=15, flip_prob=0.5))
    transform_list_train.append(RandomSkewWithFlipNP(distortion_scale=0.2, flip_prob=0.5))
    transform_list_train.append(RandomShearWithFlipNP(shear=10, flip_prob=0.5))
    transform_list_train.append(transforms.ToTensor())
    transform_list_train.append(
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    )
    transform_train = transforms.Compose(transform_list_train)

    transform_list_train_push = []
    transform_list_train_push.append(transforms.Resize(size=(img_size, img_size)))
    transform_list_train_push.append(transforms.ToTensor())
    transform_train_push = transforms.Compose(transform_list_train_push)

    transform_list_test = []
    if not preload:
        transform_list_test.append(transforms.Resize(size=(img_size, img_size)))
    transform_list_test.append(RandomRotationWithFlipNP(degrees=15, flip_prob=0.5))
    transform_list_test.append(RandomSkewWithFlipNP(distortion_scale=0.2, flip_prob=0.5))
    transform_list_test.append(RandomShearWithFlipNP(shear=10, flip_prob=0.5))
    transform_list_test.append(transforms.ToTensor())
    transform_list_test.append(
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    )
    transform_test = transforms.Compose(transform_list_test)

    # Datasets
    if img_aug:
        awa_datasets = {
            "train": AnimalDataset(
                classes_file=classes_file,
                data_path=data_path,
                img_dir_list=img_names_train,
                transform=transform_train,
                partial_predicates=partial_predicates,
                num_predicates=num_predicates,
                preload=preload,
                seed=seed,
            ),
            "train_push": AnimalDataset(
                classes_file=classes_file,
                data_path=data_path,
                img_dir_list=img_names_train,
                transform=transform_train_push,
                partial_predicates=partial_predicates,
                num_predicates=num_predicates,
                preload=False,
                seed=seed,
            ),
            "val": AnimalDataset(
                    classes_file=classes_file,
                    data_path=data_path,
                    img_dir_list=img_names_val,
                    transform=transform_train_push,
                    partial_predicates=partial_predicates,
                    num_predicates=num_predicates,
                    preload=preload,
                    seed=seed,
            ),
            "test": AnimalDataset(
                classes_file=classes_file,
                data_path=data_path,
                img_dir_list=img_names_test,
                transform=transform_train_push,
                partial_predicates=partial_predicates,
                num_predicates=num_predicates,
                preload=preload,
                seed=seed,
            ),
        }
    else:
        awa_datasets = {
            "train": AnimalDataset(
                classes_file=classes_file,
                data_path=data_path,
                img_dir_list=img_names_train,
                transform=transform_train_push,
                partial_predicates=partial_predicates,
                num_predicates=num_predicates,
                preload=preload,
                seed=seed,
            ),
            "train_push": AnimalDataset(
                classes_file=classes_file,
                data_path=data_path,
                img_dir_list=img_names_train,
                transform=transform_train_push,
                partial_predicates=partial_predicates,
                num_predicates=num_predicates,
                preload=False,
                seed=seed,
            ),
            "val": AnimalDataset(
                    classes_file=classes_file,
                    data_path=data_path,
                    img_dir_list=img_names_val,
                    transform=transform_test,
                    partial_predicates=partial_predicates,
                    num_predicates=num_predicates,
                    preload=preload,
                    seed=seed,
            ),
            "test": AnimalDataset(
                classes_file=classes_file,
                data_path=data_path,
                img_dir_list=img_names_test,
                transform=transform_test,
                partial_predicates=partial_predicates,
                num_predicates=num_predicates,
                preload=preload,
                seed=seed,
            ),
        }

    train_loader =  DataLoader(
        awa_datasets["train"],
        batch_size=batch_size,
        shuffle=False, # As per the original code
        num_workers=4,
        pin_memory=False, # As per the original code
        generator=torch.Generator().manual_seed(seed),
        drop_last=False,)

    test_loader = DataLoader(
        awa_datasets["test"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
        generator=torch.Generator().manual_seed(seed),
        drop_last=False,)

    val_loader = DataLoader(
        awa_datasets["val"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
        generator=torch.Generator().manual_seed(seed),
        drop_last=False,)

    train_push_loader = DataLoader(
        awa_datasets["train_push"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
        generator=torch.Generator().manual_seed(seed),
        drop_last=False,)

    return train_loader, test_loader, val_loader, train_push_loader