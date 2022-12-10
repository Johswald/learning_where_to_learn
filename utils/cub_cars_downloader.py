import os
import json

import numpy as np
from PIL import Image
import os
import io
import json
import glob
import h5py
import requests
from torchmeta.utils.data import Dataset, ClassDataset, CombinationMetaDataset
from torchvision.datasets.utils import check_integrity
from tqdm import tqdm
from scipy.io import loadmat
import collections
# QKFIX: See torchmeta.datasets.utils for more informations
from torchmeta.datasets.utils import get_asset

def get_asset_path(*args):
    basedir = os.path.dirname(__file__)
    print(basedir)
    return os.path.join(basedir, 'assets', *args)


def get_asset(*args, dtype=None):
    filename = get_asset_path(*args)
    if not os.path.isfile(filename):
        raise IOError('{} not found'.format(filename))

    if dtype is None:
        _, dtype = os.path.splitext(filename)
        dtype = dtype[1:]

    if dtype == 'json':
        with open(filename, 'r') as f:
            data = json.load(f)
    else:
        raise NotImplementedError()
    return data

# QKFIX: The current version of `download_file_from_google_drive` (as of torchvision==0.8.1)
# is inconsistent, and a temporary fix has been added to the bleeding-edge version of
# Torchvision. The temporary fix removes the behaviour of `_quota_exceeded`, whenever the
# quota has exceeded for the file to be downloaded. As a consequence, this means that there
# is currently no protection against exceeded quotas. If you get an integrity error in Torchmeta
# (e.g. "MiniImagenet integrity check failed" for MiniImagenet), then this means that the quota
# has exceeded for this dataset. See also: https://github.com/tristandeleu/pytorch-meta/issues/54
#
# See also: https://github.com/pytorch/vision/issues/2992
#
# The following functions are taken from
# https://github.com/pytorch/vision/blob/cd0268cd408d19d91f870e36fdffd031085abe13/torchvision/datasets/utils.py

from torchvision.datasets.utils import _get_confirm_token, _save_response_content

def _quota_exceeded(response: "requests.models.Response"):
    return False
    # See https://github.com/pytorch/vision/issues/2992 for details
    # return "Google Drive - Quota exceeded" in response.text


def download_file_from_google_drive(file_id, root, filename=None, md5=None):
    """Download a Google Drive file from  and place it in root.
    Args:
        file_id (str): id of file to be downloaded
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the id of the file.
        md5 (str, optional): MD5 checksum of the download. If None, do not check
    """
    # Based on https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
    import requests
    url = "https://docs.google.com/uc?export=download"

    root = os.path.expanduser(root)
    if not filename:
        filename = file_id
    fpath = os.path.join(root, filename)

    os.makedirs(root, exist_ok=True)

    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        session = requests.Session()

        response = session.get(url, params={'id': file_id}, stream=True)
        token = _get_confirm_token(response)

        if token:
            params = {'id': file_id, 'confirm': token}
            response = session.get(url, params=params, stream=True)

        if _quota_exceeded(response):
            msg = (
                f"The daily quota of the file {filename} is exceeded and it "
                f"can't be downloaded. This is a limitation of Google Drive "
                f"and can only be overcome by trying again later."
            )
            raise RuntimeError(msg)

        _save_response_content(response, fpath)


class CUB(CombinationMetaDataset):
    """
    The Caltech-UCSD Birds dataset, introduced in [1]. This dataset is based on
    images from 200 species of birds from the Caltech-UCSD Birds dataset [2].
    Parameters
    ----------
    root : string
        Root directory where the dataset folder `cub` exists.
    num_classes_per_task : int
        Number of classes per tasks. This corresponds to "N" in "N-way"
        classification.
    meta_train : bool (default: `False`)
        Use the meta-train split of the dataset. If set to `True`, then the
        arguments `meta_val` and `meta_test` must be set to `False`. Exactly one
        of these three arguments must be set to `True`.
    meta_val : bool (default: `False`)
        Use the meta-validation split of the dataset. If set to `True`, then the
        arguments `meta_train` and `meta_test` must be set to `False`. Exactly one
        of these three arguments must be set to `True`.
    meta_test : bool (default: `False`)
        Use the meta-test split of the dataset. If set to `True`, then the
        arguments `meta_train` and `meta_val` must be set to `False`. Exactly one
        of these three arguments must be set to `True`.
    meta_split : string in {'train', 'val', 'test'}, optional
        Name of the split to use. This overrides the arguments `meta_train`,
        `meta_val` and `meta_test` if all three are set to `False`.
    transform : callable, optional
        A function/transform that takes a `PIL` image, and returns a transformed
        version. See also `torchvision.transforms`.
    target_transform : callable, optional
        A function/transform that takes a target, and returns a transformed
        version. See also `torchvision.transforms`.
    dataset_transform : callable, optional
        A function/transform that takes a dataset (ie. a task), and returns a
        transformed version of it. E.g. `torchmeta.transforms.ClassSplitter()`.
    class_augmentations : list of callable, optional
        A list of functions that augment the dataset with new classes. These classes
        are transformations of existing classes. E.g.
        `torchmeta.transforms.HorizontalFlip()`.
    download : bool (default: `False`)
        If `True`, downloads the pickle files and processes the dataset in the root
        directory (under the `cub` folder). If the dataset is already
        available, this does not download/process the dataset again.
    Notes
    -----
    The dataset is downloaded from [2]. The dataset contains images from 200
    classes. The meta train/validation/test splits are over 100/50/50 classes.
    The splits are taken from [3] ([code](https://github.com/wyharveychen/CloserLookFewShot)
    for reproducibility).
    References
    ----------
    .. [1] Hilliard, N., Phillips, L., Howland, S., Yankov, A., Corley, C. D.,
           Hodas, N. O. (2018). Few-Shot Learning with Metric-Agnostic Conditional
           Embeddings. (https://arxiv.org/abs/1802.04376)
    .. [2] Wah, C., Branson, S., Welinder, P., Perona, P., Belongie, S. (2011).
           The Caltech-UCSD Birds-200-2011 Dataset
           (http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)
    .. [3] Chen, W., Liu, Y. and Kira, Z. and Wang, Y. and  Huang, J. (2019).
           A Closer Look at Few-shot Classification. International Conference on
           Learning Representations (https://openreview.net/forum?id=HkxLXnAcFQ)
    """
    def __init__(self, root, num_classes_per_task=None, meta_train=False,
                 meta_val=False, meta_test=False, meta_split=None,
                 transform=None, target_transform=None, dataset_transform=None,
                 class_augmentations=None, download=False):
        dataset = CUBClassDataset(root, meta_train=meta_train, meta_val=meta_val,
            meta_test=meta_test, meta_split=meta_split, transform=transform,
            class_augmentations=class_augmentations, download=download)
        super(CUB, self).__init__(dataset, num_classes_per_task,
            target_transform=target_transform, dataset_transform=dataset_transform)


class CUBClassDataset(ClassDataset):
    folder = 'cub'
    # # Google Drive ID from http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz
    gdrive_id = '1hbzc_P1FuxMkcabkgn9ZKinBwW683j45'
    tgz_filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'
    image_folder = 'CUB_200_2011/images'

    filename = '{0}_data.hdf5'
    filename_labels = '{0}_labels.json'

    def __init__(self, root, meta_train=False, meta_val=False, meta_test=False,
                 meta_split=None, transform=None, class_augmentations=None,
                 download=False):
        super(CUBClassDataset, self).__init__(meta_train=meta_train,
            meta_val=meta_val, meta_test=meta_test, meta_split=meta_split,
            class_augmentations=class_augmentations)

        self.root = os.path.join(os.path.expanduser(root), self.folder)
        self.transform = transform

        self.split_filename = os.path.join(self.root,
            self.filename.format(self.meta_split))
        self.split_filename_labels = os.path.join(self.root,
            self.filename_labels.format(self.meta_split))

        self._data_file = None
        self._data = None
        self._labels = None

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('CUB integrity check failed')
        self._num_classes = len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index % self.num_classes]
        data = self.data[label]
        transform = self.get_transform(index, self.transform)
        target_transform = self.get_target_transform(index)

        return CUBDataset(index, data, label, transform=transform,
                          target_transform=target_transform)

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def data(self):
        if self._data is None:
            self._data_file = h5py.File(self.split_filename, 'r')
            self._data = self._data_file['datasets']
        return self._data

    @property
    def labels(self):
        if self._labels is None:
            with open(self.split_filename_labels, 'r') as f:
                self._labels = json.load(f)
        return self._labels

    def _check_integrity(self):
        return (os.path.isfile(self.split_filename)
            and os.path.isfile(self.split_filename_labels))

    def close(self):
        if self._data_file is not None:
            self._data_file.close()
            self._data_file = None
            self._data = None

    def download(self):
        import tarfile
        import shutil
        import glob
        from tqdm import tqdm

        if self._check_integrity():
            return

        download_file_from_google_drive(self.gdrive_id, self.root,
            self.tgz_filename, md5=self.tgz_md5)

        tgz_filename = os.path.join(self.root, self.tgz_filename)
        with tarfile.open(tgz_filename, 'r') as f:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(f, self.root)
        image_folder = os.path.join(self.root, self.image_folder)

        for split in ['train', 'val', 'test']:
            filename = os.path.join(self.root, self.filename.format(split))
            print(filename)
            if os.path.isfile(filename):
                continue

            labels = get_asset(self.folder, '{0}.json'.format(split))
            labels_filename = os.path.join(self.root, self.filename_labels.format(split))
            with open(labels_filename, 'w') as f:
                json.dump(labels, f)

            with h5py.File(filename, 'w') as f:
                group = f.create_group('datasets')
                dtype = h5py.special_dtype(vlen=np.uint8)
                for i, label in enumerate(tqdm(labels, desc=filename)):
                    images = glob.glob(os.path.join(image_folder, label, '*.jpg'))
                    images.sort()
                    dataset = group.create_dataset(label, (len(images),), dtype=dtype)
                    for i, image in enumerate(images):
                        with open(image, 'rb') as f:
                            array = bytearray(f.read())
                            dataset[i] = np.asarray(array, dtype=np.uint8)

        tar_folder, _ = os.path.splitext(tgz_filename)
        if os.path.isdir(tar_folder):
            shutil.rmtree(tar_folder)

        attributes_filename = os.path.join(self.root, 'attributes.txt')
        if os.path.isfile(attributes_filename):
            os.remove(attributes_filename)


class CUBDataset(Dataset):
    def __init__(self, index, data, label,
                 transform=None, target_transform=None):
        super(CUBDataset, self).__init__(index, transform=transform,
                                         target_transform=target_transform)
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = Image.open(io.BytesIO(self.data[index])).convert('RGB')
        target = self.label

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (image, target)


class CARS(CombinationMetaDataset):
    def __init__(self, root, num_classes_per_task=None, meta_train=False,
                 meta_val=False, meta_test=False, meta_split=None,
                 transform=None, target_transform=None, dataset_transform=None,
                 class_augmentations=None, download=False):
        dataset = CARSClassDataset(root, meta_train=meta_train,
            meta_val=meta_val, meta_test=meta_test, meta_split=meta_split,
            transform=transform, class_augmentations=class_augmentations,
            download=download)
        super(CARS, self).__init__(dataset, num_classes_per_task,
            target_transform=target_transform, dataset_transform=dataset_transform)

class CARSClassDataset(ClassDataset):
    folder = 'cars'

    train_tar_url = 'http://ai.stanford.edu/~jkrause/car196/cars_train.tgz'
    test_tar_url = 'http://ai.stanford.edu/~jkrause/car196/cars_test.tgz'
    devkit_tar_url = 'https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz'

    filename = '{0}_data.hdf5'
    filename_labels = '{0}_labels.json'

    def __init__(self, root, meta_train=False, meta_val=False, meta_test=False,
                 meta_split=None, transform=None, class_augmentations=None,
                 download=False):
        super(CARSClassDataset, self).__init__(meta_train=meta_train,
            meta_val=meta_val, meta_test=meta_test, meta_split=meta_split,
            class_augmentations=class_augmentations)

        self.root = os.path.join(os.path.expanduser(root), self.folder)
        self.transform = transform

        self.split_filename = os.path.join(self.root,
            self.filename.format(self.meta_split))
        self.split_filename_labels = os.path.join(self.root,
            self.filename_labels.format(self.meta_split))

        self._data_file = None
        self._data = None
        self._labels = None

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('CARS integrity check failed')
        self._num_classes = len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index % self.num_classes]
        data = self.data[label]
        transform = self.get_transform(index, self.transform)
        target_transform = self.get_target_transform(index)

        return CARSDataset(index, data, label,
            transform=transform, target_transform=target_transform)

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def data(self):
        if self._data is None:
            self._data_file = h5py.File(self.split_filename, 'r')
            self._data = self._data_file['datasets']
        return self._data

    @property
    def labels(self):
        if self._labels is None:
            with open(self.split_filename_labels, 'r') as f:
                self._labels = json.load(f)
        return self._labels

    def _check_integrity(self):
        return (os.path.isfile(self.split_filename)
            and os.path.isfile(self.split_filename_labels))

    def close(self):
        if self._data_file is not None:
            self._data_file.close()
            self._data_file = None
            self._data = None

    def download(self):
        import tarfile

        if self._check_integrity():
            return

        chunkSize = 1024
        r = requests.get(self.train_tar_url, stream=True)
        print("done")
        with open(self.root+'/cars_train.tgz', 'wb') as f:
            pbar = tqdm( unit="B", total=int( r.headers['Content-Length'] ) )
            for chunk in r.iter_content(chunk_size=chunkSize):
                if chunk: # filter out keep-alive new chunks
                    pbar.update (len(chunk))
                    f.write(chunk)

        r = requests.get(self.devkit_tar_url, stream=True)
        with open(self.root+'/car_devkit.tgz', 'wb') as f:
            pbar = tqdm( unit="B", total=int( r.headers['Content-Length'] ) )
            for chunk in r.iter_content(chunk_size=chunkSize):
                if chunk: # filter out keep-alive new chunks
                    pbar.update (len(chunk))
                    f.write(chunk)

        filename = os.path.join(self.root, 'cars_train.tgz')
        with tarfile.open(filename, 'r') as f:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(f, self.root)

        filename = os.path.join(self.root, 'car_devkit.tgz')
        with tarfile.open(filename, 'r') as f:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(f, self.root)

        annos_path = os.path.join(self.root, 'devkit', 'cars_train_annos.mat')
        cars_meta_path = os.path.join(self.root, 'devkit', 'cars_meta.mat')

        annos = loadmat(annos_path)['annotations'][0]
        cars_meta = loadmat(cars_meta_path)['class_names'][0]
        cars_meta = [c[0] for c in cars_meta]

        names_to_bboxes = {}
        clss_to_names = collections.defaultdict(list)

        for xmin, ymin, xmax, ymax, label, filename in annos:
            bbox = (int(xmin[0][0]), int(ymin[0][0]), int(xmax[0][0]), int(ymax[0][0]))
            label = int(label[0][0]) - 1
            filename = str(filename[0])
            names_to_bboxes[filename] = bbox
            clss_to_names[cars_meta[label]].append(filename)

        for split in ['train', 'val', 'test']:
            filename = os.path.join(self.root, self.filename.format(split))
            labels = get_asset(self.folder, '{}.json'.format(split))
            labels_filename = os.path.join(self.root, self.filename_labels.format(split))

            with open(labels_filename, 'w') as f:
                json.dump(labels, f)

            with h5py.File(filename, 'w') as f:
                group = f.create_group('datasets')
                for i, label in enumerate(tqdm(labels, desc=filename)):
                    images = []
                    for file in clss_to_names[label]:
                        file_path = os.path.join(self.root,
                                                'cars_train',
                                                 file)
                        img = Image.open(file_path).convert('RGB')
                        bbox = names_to_bboxes[file]
                        img = np.asarray(img.crop(bbox).resize((84, 84)), dtype=np.uint8)
                        images.append(img)

                    dataset = group.create_dataset(label, (len(images), 84, 84, 3))

                    for j, image in enumerate(images):
                        dataset[j] = image

class CARSDataset(Dataset):
    def __init__(self, index, data, label,
                 transform=None, target_transform=None):
        super(CARSDataset, self).__init__(index, transform=transform,
                                                  target_transform=target_transform)
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = Image.fromarray(self.data[index].astype(np.uint8)).convert('RGB')
        target = self.label

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (image, target)