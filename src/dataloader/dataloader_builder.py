from torch.utils.data import DataLoader
import torchvision.transforms as standard_transforms

from utils.transformations.composition import ComposeSemSeg, ComposeObjDet, ComposeResize
from utils.transformations.crop import CropSegSem, CropObjDet
from utils.transformations.flip import RandomHorizontalFlipSegSem, RandomHorizontalFlipObjDet
from utils.transformations.input_preprocess import PreprocessInput
from utils.transformations.random_distort import RandomDistort
from utils.transformations.resize import Resize
from utils.transformations.tensor import ToTensor

from .from_file_dataset_classification import FromFileDatasetClassification
from .from_file_dataset_detection import FromFileDatasetDetection
from .from_file_dataset_segmentation import FromFileDatasetSegmentation
from .from_file_dataset_to_predict import FromFileDatasetToPredict
from .from_dataset_structure import FromDatasetStructure


class DataLoaderBuilder:
    def __init__(self, cf, model):
        self.cf = cf
        self.model = model

        self.train_set = None
        self.train_loader = None
        self.valid_set = None
        self.valid_loader = None
        self.predict_set = None
        self.predict_loader = None

        # Compose preprocesing function for dataloaders
        if self.cf.problem_type == 'detection':
            self.img_preprocessing = standard_transforms.Compose([
                RandomDistort(self.cf),
                PreprocessInput(self.cf),
                standard_transforms.ToTensor()
            ])
            self.train_transformation = ComposeObjDet([
                CropObjDet(self.cf),
                RandomHorizontalFlipObjDet(self.cf)
            ])
            self.resize = ComposeResize([Resize(self.cf)])
        else:
            self.img_preprocessing = standard_transforms.Compose([
                RandomDistort(self.cf),
                PreprocessInput(self.cf),
                ToTensor()
            ])
            self.train_transformation = ComposeSemSeg([
                CropSegSem(self.cf),
                RandomHorizontalFlipSegSem(self.cf)
            ])

    def build_train(self):
        if self.cf.problem_type == 'segmentation':
            self.train_set = FromFileDatasetSegmentation(self.cf, self.cf.train_images_txt, self.cf.train_gt_txt,
                                                         self.cf.train_samples, self.cf.resize_image_train,
                                                         preprocess=self.img_preprocessing,
                                                         transform=self.train_transformation)
        elif self.cf.problem_type == 'classification':
            if self.cf.train_dataset_path is not None:
                self.train_set = FromDatasetStructure(self.cf, self.cf.train_dataset_path, self.cf.resize_image_train,
                                                      preprocess=self.img_preprocessing,
                                                      transform=self.train_transformation)
            else:
                self.train_set = FromFileDatasetClassification(self.cf, self.cf.train_images_txt, self.cf.train_gt_txt,
                                                               self.cf.train_samples, self.cf.resize_image_train,
                                                               preprocess=self.img_preprocessing,
                                                               transform=self.train_transformation)
        elif self.cf.problem_type == 'detection':
            self.train_set = FromFileDatasetDetection(self.cf, self.cf.train_images_txt, self.cf.train_gt_txt,
                                                      self.cf.train_samples, self.cf.resize_image_train,
                                                      preprocess=self.img_preprocessing,
                                                      transform=self.train_transformation,
                                                      box_coder=self.model.box_coder,
                                                      resize_process=self.resize)
        self.train_loader = DataLoader(self.train_set, batch_size=self.cf.train_batch_size, num_workers=4)

    def build_valid(self, valid_samples, images_txt, gt_txt, resize_image, batch_size):
        if self.cf.problem_type == 'segmentation':
            self.valid_set = FromFileDatasetSegmentation(self.cf, images_txt, gt_txt,
                                                         valid_samples, resize_image,
                                                         preprocess=self.img_preprocessing, transform=None,
                                                         valid=True)
        elif self.cf.problem_type == 'classification':
            if self.cf.valid_dataset_path is not None:
                self.valid_set = FromDatasetStructure(self.cf, self.cf.valid_dataset_path, resize_image,
                                                      preprocess=self.img_preprocessing,
                                                      transform=None, valid=True)
            else:
                self.valid_set = FromFileDatasetClassification(self.cf, images_txt, gt_txt,
                                                               valid_samples, resize_image,
                                                               preprocess=self.img_preprocessing, transform=None,
                                                               valid=True)
        elif self.cf.problem_type == 'detection':
            self.train_transformation = ComposeObjDet([Resize(self.cf)])
            self.valid_set = FromFileDatasetDetection(self.cf, images_txt, gt_txt,
                                                      valid_samples, resize_image,
                                                      preprocess=self.img_preprocessing,
                                                      transform=None,
                                                      valid=True,
                                                      resize_process=self.resize)
        self.valid_loader = DataLoader(self.valid_set, batch_size=batch_size, num_workers=4)

    def build_predict(self):
        if self.cf.problem_type == 'classification' and self.cf.test_dataset_path is not None:
            self.predict_set = FromDatasetStructure(self.cf, self.cf.test_dataset_path, self.cf.resize_image_test,
                                                  preprocess=self.img_preprocessing,)
        else:
            self.predict_set = FromFileDatasetToPredict(self.cf, self.cf.test_images_txt,
                                                        self.cf.test_samples, self.cf.resize_image_test,
                                                        preprocess=self.img_preprocessing)

        self.predict_loader = DataLoader(self.predict_set, batch_size=1, num_workers=4)
