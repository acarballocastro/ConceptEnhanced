import pickle
import numpy as np
import os
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from glob import glob
from tqdm import tqdm

imgs = {}

class AnimalDataset(Dataset):
    """
    Animals with attributes dataset class
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

def get_AwA_dataloaders(
    classes_file,
    data_path,
    batch_size,
    num_workers,
    img_size=224,
    seed=42,
    partial_predicates: bool = False,
    num_predicates: int = 85,
    preload: bool = True,
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
    # Transformations
    transform_list_train = []
    if not preload:
        transform_list_train.append(transforms.Resize(size=(img_size, img_size)))
    transform_list_train.append(transforms.ToTensor())
    transform_list_train.append(
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    )
    transform_train = transforms.Compose(transform_list_train)

    # Datasets
    awa_dataset = AnimalDataset(
            classes_file=classes_file,
            data_path=data_path,
            img_dir_list=None,
            transform=transform_train,
            partial_predicates=False,
            num_predicates=85,
            preload=True,
            seed=seed)

    # Sampling half of the dataset
    dataset_size = len(awa_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.2 * dataset_size))
    
    np.random.seed(seed)
    np.random.shuffle(indices)
    
    subset_indices = indices[:split]
    sampler = SubsetRandomSampler(subset_indices)

    dataloader = DataLoader(awa_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)

    return dataloader