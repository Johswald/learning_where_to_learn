import os
from torchmeta.datasets import Omniglot, MiniImagenet, TieredImagenet
from torchmeta.transforms import ClassSplitter, Categorical
from torchvision.transforms import ToTensor, Resize, Compose
from torchvision import transforms

from torchmeta.utils.data import BatchMetaDataLoader
from utils.cub_cars_downloader import CUB, CARS

def list_to_str(list_arg, delim=' '):
    """Convert a list of numbers into a string.

    Args:
        list_arg: List of numbers.
        delim (optional): Delimiter between numbers.

    Returns:
        List converted to string.
    """
    ret = ''
    for i, e in enumerate(list_arg):
        if i > 0:
            ret += delim
        ret += str(e)
    return ret

def get_subdict(adict, name):
    if adict is None:
        return adict
    tmp = {k[len(name) + 1:]:adict[k] for k in adict if name in k}
    return tmp

def load_data(args):
    meta_dataloader={}
    #-------------Load data----------------------
    if args.dataset=="MiniImagenet":
        dataset_transform = ClassSplitter(shuffle=True,
                                    num_train_per_class=args.num_shots_train,
                                    num_test_per_class=args.num_shots_test)
        
        if args.data_aug:            
            transform_train = Compose([
                    Resize(84),
                    transforms.RandomCrop(84, padding=8),
                    transforms.ColorJitter(brightness=0.4, 
                                           contrast=0.4, 
                                           saturation=0.4),
                    transforms.RandomHorizontalFlip(),  
                    transforms.ToTensor(), 
                    transforms.Normalize((0.485, 0.456, 0.406), 
                                         (0.229, 0.224, 0.225))])
        else:
            transform_train = Compose([
                Resize(84),
                transforms.ToTensor(), 
                transforms.Normalize((0.485, 0.456, 0.406), 
                                     (0.229, 0.224, 0.225))])

        transform_test = Compose([
                Resize(84),
                transforms.ToTensor(), 
                transforms.Normalize((0.485, 0.456, 0.406), 
                                     (0.229, 0.224, 0.225))])

        meta_train_dataset = MiniImagenet("data",
                                    transform=transform_train,
                                    target_transform=Categorical(args.num_ways),
                                    num_classes_per_task=args.num_ways,
                                    meta_train=True,
                                    dataset_transform=dataset_transform,
                                    download=True)
                                          
        meta_val_dataset = MiniImagenet("data",
                                    transform=transform_test,
                                    target_transform=Categorical(args.num_ways),
                                    num_classes_per_task=args.num_ways,
                                    meta_val=True,
                                    dataset_transform=dataset_transform)

        meta_test_dataset = MiniImagenet("data",
                                    transform=transform_test,
                                    target_transform=Categorical(args.num_ways),
                                    num_classes_per_task=args.num_ways,
                                    meta_test=True,
                                    dataset_transform=dataset_transform)


        meta_dataloader["train"] = BatchMetaDataLoader(meta_train_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                num_workers=args.num_workers,
                                                pin_memory=True)
                                                
        meta_dataloader["val"]=BatchMetaDataLoader(meta_val_dataset,
                                                batch_size=args.test_batch_size,
                                                shuffle=True,
                                                num_workers=args.num_workers,
                                                pin_memory=True)

        meta_dataloader["test"]=BatchMetaDataLoader(meta_test_dataset,
                                                batch_size=args.test_batch_size,
                                                shuffle=True,
                                                num_workers=args.num_workers,
                                                pin_memory=True)

        feature_size=5*5*args.hidden_size
        input_channels=3

    if args.dataset=="Omniglot":
        print("Omniglot not supported.")
        exit()

    if args.dataset=="TieredImagenet":
        dataset_transform = ClassSplitter(shuffle=True,
                                    num_train_per_class=args.num_shots_train,
                                    num_test_per_class=args.num_shots_test)
        test_transform = Compose([Resize(84), ToTensor()])
        train_transform = Compose([Resize(84), ToTensor()])

        meta_train_dataset = TieredImagenet("data",
                                    transform=train_transform,
                                    target_transform=Categorical(args.num_ways),
                                    num_classes_per_task=args.num_ways,
                                    meta_train=True,
                                    dataset_transform=dataset_transform,
                                    download=True)

        meta_val_dataset = TieredImagenet("data",
                                    transform=test_transform,
                                    target_transform=Categorical(args.num_ways),
                                    num_classes_per_task=args.num_ways,
                                    meta_val=True,
                                    dataset_transform=dataset_transform)

        meta_test_dataset = TieredImagenet("data",
                                    transform=test_transform,
                                    target_transform=Categorical(args.num_ways),
                                    num_classes_per_task=args.num_ways,
                                    meta_test=True,
                                    dataset_transform=dataset_transform)


        meta_dataloader["train"] = BatchMetaDataLoader(meta_train_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                num_workers=args.num_workers,
                                                pin_memory=True)

        meta_dataloader["val"]=BatchMetaDataLoader(meta_val_dataset,
                                                batch_size=args.test_batch_size,
                                                shuffle=True,
                                                num_workers=args.num_workers,
                                                pin_memory=True)

        meta_dataloader["test"]=BatchMetaDataLoader(meta_test_dataset,
                                                batch_size=args.test_batch_size,
                                                shuffle=True,
                                                num_workers=args.num_workers,
                                                pin_memory=True)

        feature_size=5*5*args.hidden_size
        input_channels=3

    if args.dataset=="CUB":
        dataset_transform = ClassSplitter(shuffle=True,
                                    num_train_per_class=args.num_shots_train,
                                    num_test_per_class=args.num_shots_test)
        test_transform = Compose([Resize((84,84)), ToTensor()])
        train_transform = Compose([Resize((84,84)), ToTensor()])

        meta_train_dataset = CUB("data",
                                  transform=train_transform,
                                  target_transform=Categorical(args.num_ways),
                                  num_classes_per_task=args.num_ways,
                                  meta_train=True,
                                  dataset_transform=dataset_transform,
                                  download=True)

        meta_val_dataset = CUB("data",
                                transform=test_transform,
                                target_transform=Categorical(args.num_ways),
                                num_classes_per_task=args.num_ways,
                                meta_val=True,
                                dataset_transform=dataset_transform)

        meta_test_dataset = CUB("data",
                                  transform=test_transform,
                                  target_transform=Categorical(args.num_ways),
                                  num_classes_per_task=args.num_ways,
                                  meta_test=True,
                                  dataset_transform=dataset_transform)


        meta_dataloader["train"] = BatchMetaDataLoader(meta_train_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                num_workers=args.num_workers,
                                                pin_memory=True)

        meta_dataloader["val"]=BatchMetaDataLoader(meta_val_dataset,
                                                batch_size=args.test_batch_size,
                                                shuffle=True,
                                                num_workers=args.num_workers,
                                                pin_memory=True)

        meta_dataloader["test"]=BatchMetaDataLoader(meta_test_dataset,
                                                batch_size=args.test_batch_size,
                                                shuffle=True,
                                                num_workers=args.num_workers,
                                                pin_memory=True)

        feature_size=5*5*args.hidden_size
        input_channels=3

    if args.dataset=="CARS":
        
        if not os.path.exists("data/cars"):
            os.makedirs("data/cars")

        dataset_transform = ClassSplitter(shuffle=True,
                                    num_train_per_class=args.num_shots_train,
                                    num_test_per_class=args.num_shots_test)
        test_transform = Compose([Resize((84,84)), ToTensor()])
        train_transform = Compose([Resize((84,84)), ToTensor()])

        meta_train_dataset = CARS("data",
                                  transform=train_transform,
                                  target_transform=Categorical(args.num_ways),
                                  num_classes_per_task=args.num_ways,
                                  meta_train=True,
                                  dataset_transform=dataset_transform,
                                  download=True)

        meta_val_dataset = CARS("data",
                                transform=test_transform,
                                target_transform=Categorical(args.num_ways),
                                num_classes_per_task=args.num_ways,
                                meta_val=True,
                                dataset_transform=dataset_transform)

        meta_test_dataset = CARS("data",
                                  transform=test_transform,
                                  target_transform=Categorical(args.num_ways),
                                  num_classes_per_task=args.num_ways,
                                  meta_test=True,
                                  dataset_transform=dataset_transform)


        meta_dataloader["train"] = BatchMetaDataLoader(meta_train_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                num_workers=args.num_workers,
                                                pin_memory=True)

        meta_dataloader["val"]=BatchMetaDataLoader(meta_val_dataset,
                                                batch_size=args.test_batch_size,
                                                shuffle=True,
                                                num_workers=args.num_workers,
                                                pin_memory=True)

        meta_dataloader["test"]=BatchMetaDataLoader(meta_test_dataset,
                                                batch_size=args.test_batch_size,
                                                shuffle=True,
                                                num_workers=args.num_workers,
                                                pin_memory=True)

        feature_size=5*5*args.hidden_size
        input_channels=3

    return meta_dataloader, feature_size, input_channels


