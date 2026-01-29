import sys

from PIL import Image

from torchvision import datasets, transforms
from base import BaseDataLoader
from data_loader.cifar10 import get_cifar10
from data_loader.cifar100 import get_cifar100
from data_loader.clothing1m import get_clothing1m
from data_loader.clothing1m2 import get_clothing
from parse_config import ConfigParser


from data_loader.augmentations import Augmentation, CutoutDefault
from data_loader.augmentation_archive import (
    autoaug_policy,
    autoaug_paper_cifar10,
    fa_reduced_cifar10,
)  # , autoaug_imagenet_policy, svhn_policies


class CIFAR10DataLoader(BaseDataLoader):
    def __init__(
        self,
        data_dir,
        batch_size,
        shuffle=True,
        validation_split=0.0,
        num_batches=0,
        training=True,
        num_workers=4,
        pin_memory=True,
    ):
        config = ConfigParser.get_instance()
        cfg_trainer = config["trainer"]

        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        transform_val = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        if config["train_loss"]["args"]["ratio_consistency"] > 0:
            transform_train_aug = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            )

            if config["data_augmentation"]["type"] is not None:

                autoaug = transforms.Compose([])
                # if isinstance(cfg_trainer['aug'], list):
                #     autoaug.transforms.insert(0, Augmentation(C.get()['aug']))
                # else:
                if config["data_augmentation"]["type"] == "fa_reduced_cifar10":
                    autoaug.transforms.insert(0, Augmentation(fa_reduced_cifar10()))
                elif config["data_augmentation"]["type"] == "autoaug_cifar10":
                    autoaug.transforms.insert(0, Augmentation(autoaug_paper_cifar10()))
                elif config["data_augmentation"]["type"] == "autoaug_extend":
                    autoaug.transforms.insert(0, Augmentation(autoaug_policy()))
                elif config["data_augmentation"]["type"] == "default":
                    pass
                else:
                    raise ValueError(
                        "not found augmentations. %s"
                        % config["data_augmentation"]["type"]
                    )
                transform_train_aug.transforms.insert(0, autoaug)

                if config["data_augmentation"]["cutout"] > 0:
                    transform_train_aug.transforms.append(
                        CutoutDefault(config["data_augmentation"]["cutout"])
                    )
        else:
            transform_train_aug = None

        self.data_dir = data_dir

        noise_file = "%sCIFAR10_%.1f_Asym_%s.json" % (
            config["data_loader"]["args"]["data_dir"],
            cfg_trainer["percent"],
            cfg_trainer["asym"],
        )

        self.train_dataset, self.val_dataset, noise_level = get_cifar10(
            config["data_loader"]["args"]["data_dir"],
            cfg_trainer,
            train=training,
            transform_train=transform_train,
            transform_train_aug=transform_train_aug,
            transform_val=transform_val,
            noise_file=noise_file,
            download=True,
        )
        print(f"noise_level: {noise_level:.2%}")
        self.noise_level = noise_level

        super().__init__(
            self.train_dataset,
            batch_size,
            shuffle,
            validation_split,
            num_workers,
            pin_memory,
            val_dataset=self.val_dataset,
        )

    # def run_loader(self, batch_size, shuffle, validation_split, num_workers, pin_memory):
    #     super().__init__(self.train_dataset, batch_size, shuffle, validation_split, num_workers, pin_memory,
    #                      val_dataset = self.val_dataset)

    def run_loader(self):
        pass


class CIFAR100DataLoader(BaseDataLoader):
    def __init__(
        self,
        data_dir,
        batch_size,
        shuffle=True,
        validation_split=0.0,
        num_batches=0,
        training=True,
        num_workers=4,
        pin_memory=True,
    ):
        config = ConfigParser.get_instance()
        cfg_trainer = config["trainer"]

        transform_train = transforms.Compose(
            [
                # transforms.ColorJitter(brightness= 0.4, contrast= 0.4, saturation= 0.4, hue= 0.1),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                ),
            ]
        )
        transform_val = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                ),
            ]
        )
        transform_train_aug = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        if config["data_augmentation"]["type"] is not None:

            autoaug = transforms.Compose([])
            # if isinstance(cfg_trainer['aug'], list):
            #     autoaug.transforms.insert(0, Augmentation(C.get()['aug']))
            # else:
            if config["data_augmentation"]["type"] == "fa_reduced_cifar10":
                autoaug.transforms.insert(0, Augmentation(fa_reduced_cifar10()))
            elif config["data_augmentation"]["type"] == "autoaug_cifar10":
                autoaug.transforms.insert(0, Augmentation(autoaug_paper_cifar10()))
            elif config["data_augmentation"]["type"] == "autoaug_extend":
                autoaug.transforms.insert(0, Augmentation(autoaug_policy()))
            elif config["data_augmentation"]["type"] == "default":
                pass
            else:
                raise ValueError(
                    "not found augmentations. %s" % config["data_augmentation"]["type"]
                )
            transform_train_aug.transforms.insert(0, autoaug)
            # transform_train.transforms.insert(0, autoaug)

            if config["data_augmentation"]["cutout"] > 0:
                transform_train_aug.transforms.append(
                    CutoutDefault(config["data_augmentation"]["cutout"])
                )

        self.data_dir = data_dir
        config = ConfigParser.get_instance()

        noise_file = "%sCIFAR100_%.1f_Asym_%s.json" % (
            config["data_loader"]["args"]["data_dir"],
            cfg_trainer["percent"],
            cfg_trainer["asym"],
        )

        self.train_dataset, self.val_dataset, noise_level = get_cifar100(
            config["data_loader"]["args"]["data_dir"],
            cfg_trainer,
            train=training,
            transform_train=transform_train,
            transform_train_aug=transform_train_aug,
            transform_val=transform_val,
            noise_file=noise_file,
        )
        print(f"noise_level: {noise_level:.2%}")
        self.noise_level = noise_level
        super().__init__(
            self.train_dataset,
            batch_size,
            shuffle,
            validation_split,
            num_workers,
            pin_memory,
            val_dataset=self.val_dataset,
        )

    # def run_loader(self, batch_size, shuffle, validation_split, num_workers, pin_memory):
    #     super().__init__(self.train_dataset, batch_size, shuffle, validation_split, num_workers, pin_memory,
    #                      val_dataset = self.val_dataset)

    def run_loader(self):
        pass


class CLOTHING1MDataLoader(BaseDataLoader):
    def __init__(
        self,
        data_dir,
        batch_size,
        shuffle=True,
        validation_split=0.0,
        num_batches=0,
        training=True,
        num_workers=4,
        pin_memory=True,
    ):
        config = ConfigParser.get_instance()

        transform_train = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        transform_val = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        if config["train_loss"]["args"]["ratio_consistency"] > 0:
            transform_train_aug = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.RandomCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )

            if config["data_augmentation"]["type"] is not None:
                autoaug = transforms.Compose([])
                if config["data_augmentation"]["type"] == "autoaug_extend":
                    autoaug.transforms.insert(0, Augmentation(autoaug_policy()))
                elif config["data_augmentation"]["type"] == "default":
                    pass
                else:
                    raise ValueError(
                        "not found augmentations. %s"
                        % config["data_augmentation"]["type"]
                    )
                transform_train_aug.transforms.insert(0, autoaug)

                if config["data_augmentation"]["cutout"] > 0:
                    transform_train_aug.transforms.append(
                        CutoutDefault(config["data_augmentation"]["cutout"])
                    )
        else:
            transform_train_aug = None

        self.data_dir = data_dir

        self.train_dataset, self.val_dataset, noise_level = get_clothing1m(
            data_dir,
            train=training,
            transform_train=transform_train,
            transform_train_aug=transform_train_aug,
            transform_val=transform_val,
        )

        print(f"noise_level: {noise_level:.2%}")
        print(f"len dataset: {len(self.train_dataset)}")
        self.noise_level = noise_level

        super().__init__(
            self.train_dataset,
            batch_size,
            shuffle,
            validation_split,
            num_workers,
            pin_memory,
            val_dataset=self.val_dataset,
        )

    def run_loader(self):
        pass


class Clothing1MDataLoaderELR(BaseDataLoader):
    # copy paste from ELR
    def __init__(
        self,
        data_dir,
        batch_size,
        shuffle=True,
        validation_split=0.0,
        num_batches=0,
        training=True,
        num_workers=4,
        pin_memory=True,
    ):
        config = ConfigParser.get_instance()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_batches = num_batches
        self.training = training

        self.transform_train = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)
                ),
            ]
        )
        if config["train_loss"]["args"]["ratio_consistency"] > 0:
            transform_train_aug = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.RandomCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)
                    ),
                ]
            )

            if config["data_augmentation"]["type"] is not None:

                autoaug = transforms.Compose([])
                # if isinstance(cfg_trainer['aug'], list):
                #     autoaug.transforms.insert(0, Augmentation(C.get()['aug']))
                # else:
                if config["data_augmentation"]["type"] == "fa_reduced_cifar10":
                    autoaug.transforms.insert(0, Augmentation(fa_reduced_cifar10()))
                elif config["data_augmentation"]["type"] == "autoaug_cifar10":
                    autoaug.transforms.insert(0, Augmentation(autoaug_paper_cifar10()))
                elif config["data_augmentation"]["type"] == "autoaug_extend":
                    autoaug.transforms.insert(0, Augmentation(autoaug_policy()))
                elif config["data_augmentation"]["type"] == "default":
                    pass
                else:
                    raise ValueError(
                        "not found augmentations. %s"
                        % config["data_augmentation"]["type"]
                    )
                transform_train_aug.transforms.insert(0, autoaug)

                if config["data_augmentation"]["cutout"] > 0:
                    transform_train_aug.transforms.append(
                        CutoutDefault(config["data_augmentation"]["cutout"])
                    )
        else:
            transform_train_aug = None
        self.transform_train_aug = transform_train_aug

        self.transform_val = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)
                ),
            ]
        )

        cfg_trainer = config["trainer"]
        self.train_dataset, self.val_dataset = get_clothing(
            config["data_loader"]["args"]["data_dir"],
            cfg_trainer,
            num_samples=self.num_batches * self.batch_size,
            train=training,
            transform_train=self.transform_train,
            transform_train_aug=self.transform_train_aug,
            transform_val=self.transform_val,
        )

        super().__init__(
            self.train_dataset,
            batch_size,
            shuffle,
            validation_split,
            num_workers,
            pin_memory,
            val_dataset=self.val_dataset,
        )
        self.noise_level = 0.38267061364909194
