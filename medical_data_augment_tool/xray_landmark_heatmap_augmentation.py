from medical_data_augment_tool.data_augmentation_base import MedicalDataAugmentationBase

import numpy as np
import SimpleITK as sitk

from medical_data_augment_tool.transformations.intensity.np.shift_scale_clamp import ShiftScaleClamp
from medical_data_augment_tool.transformations.spatial import translation, scale, composite, rotation, deformation
from medical_data_augment_tool.transformations.intensity.np.normalize import normalize_robust
from medical_data_augment_tool.utils.sitk_image import reduce_dimension
from medical_data_augment_tool.generators.landmark_generator import LandmarkGeneratorHeatmap
from medical_data_augment_tool.generators.landmark_generator import LandmarkGenerator
from medical_data_augment_tool.datasources.landmark_datasource import LandmarkDataSource
from medical_data_augment_tool.generators.image_generator import ImageGenerator
from medical_data_augment_tool.datasources.cached_image_datasource import CachedImageDataSource
from medical_data_augment_tool.datasources.image_datasource import ImageDataSource
from medical_data_augment_tool.iterators.id_list_iterator import IdListIterator
from medical_data_augment_tool.datasets.reference_image_transformation_dataset import ReferenceTransformationDataset


def spatial_transformation(image_size, dim=2):
    """
    The spatial image transformation without random augmentation.
    :return: The transformation.
    """
    return composite.Composite(dim,
                               [translation.InputCenterToOrigin(dim),
                                scale.FitFixedAr(dim, image_size),
                                translation.OriginToOutputCenter(dim, image_size)]
                               )


class XrayDataAugmentationLegacy(MedicalDataAugmentationBase):

    def __init__(self, config_dic: {}, image_base_folder, landmarks_file_path, cv_file_path, train: bool) -> None:
        super().__init__()
        self.image_base_folder = image_base_folder
        self.landmarks_file_path = landmarks_file_path
        self.cv_file_path = cv_file_path
        self.train = train

        # settings
        self.image_size = config_dic["input_size_model"]
        self.heatmap_size = config_dic["output_size_model"]
        self.sigma = config_dic["heatmap_blob_sigma"]
        self.num_landmarks = config_dic["num_landmarks"]
        self.data_format = config_dic["data_format"]
        self.save_debug_images = False
        self.heatmap_scale = config_dic["heatmap_scale"]
        self.dim = config_dic["dim"]
        self.input_image_ext = config_dic["input_image_ext"]

        self.downsampling_factor = self.image_size[0] / self.heatmap_size[0]

        if self.train:
            self.dataset = self.dataset_train()
        else:
            self.dataset = self.dataset_val()

    def spatial_transformation_augmented(self):
        """
        The spatial image transformation with random augmentation.
        :return: The transformation.
        """
        return composite.Composite(self.dim,
                                   [translation.InputCenterToOrigin(self.dim),
                                    scale.FitFixedAr(self.dim, self.image_size),
                                    translation.Random(self.dim, [10, 10]),
                                    rotation.Random(self.dim, [0.2, 0.2]),
                                    translation.OriginToOutputCenter(self.dim, self.image_size),
                                    deformation.Output(self.dim, [5, 5], 20, self.image_size)
                                    ]
                                   )

    def intensity_postprocessing_augmented(self, image):
        """
        Intensity postprocessing. Random augmentation version.
        :param image: The np input image.
        :return: The processed image.
        """
        normalized = normalize_robust(image)
        return ShiftScaleClamp(random_shift=0.15,
                               random_scale=0.15)(normalized)

    def intensity_postprocessing(self, image):
        """
        Intensity postprocessing.
        :param image: The np input image.
        :return: The processed image.
        """
        normalized = normalize_robust(image)
        return normalized

    def data_sources(self, cached, image_extension='.mhd'):
        """
        Returns the data sources that load data.
        {
        'image_datasource:' ImageDataSource that loads the image files.
        'landmarks_datasource:' LandmarkDataSource that loads the landmark coordinates.
        }
        :param cached: If true, use a CachedImageDataSource instead of an ImageDataSource.
        :param image_extension: The image extension of the input data.
        :return: A dict of data sources.
        """
        if cached:
            image_datasource = CachedImageDataSource(self.image_base_folder,
                                                     '',
                                                     '',
                                                     image_extension,
                                                     preprocessing=reduce_dimension,
                                                     set_identity_spacing=True,
                                                     cache_maxsize=16384)
        else:
            image_datasource = ImageDataSource(self.image_base_folder,
                                               '',
                                               '',
                                               image_extension,
                                               preprocessing=reduce_dimension,
                                               set_identity_spacing=True)
        landmarks_datasource = LandmarkDataSource(self.landmarks_file_path,
                                                  self.num_landmarks,
                                                  self.dim)
        return {'image_datasource': image_datasource,
                'landmarks_datasource': landmarks_datasource}

    def data_generators(self, image_post_processing_np):
        """
        Returns the data generators that process one input. See datasources() for dict values.
        :param image_post_processing_np: The np postprocessing function for the image data generator.
        :return: A dict of data generators.
        """
        image_generator = ImageGenerator(self.dim,
                                         self.image_size,
                                         post_processing_np=image_post_processing_np,
                                         interpolator='linear',
                                         resample_default_pixel_value=0,
                                         data_format=self.data_format,
                                         resample_sitk_pixel_type=sitk.sitkFloat32,
                                         np_pixel_type=np.float32)
        if self.downsampling_factor == 1:
            heatmap_post_transformation = None
        else:
            heatmap_post_transformation = scale.Fixed(self.dim, self.downsampling_factor)
        landmark_heatmap_generator = LandmarkGeneratorHeatmap(self.dim,
                                                              self.heatmap_size,
                                                              [1] * self.dim,
                                                              self.sigma,
                                                              scale_factor=self.heatmap_scale,
                                                              normalize_center=True,
                                                              data_format=self.data_format,
                                                              post_transformation=heatmap_post_transformation)
        landmark_2Dpoints_generator = LandmarkGenerator(self.dim,
                                                        self.heatmap_size,
                                                        [1] * self.dim)
        return {'image': image_generator,
                'landmarks_heatmap': landmark_heatmap_generator,
                'landmarks_2Dpoints': landmark_2Dpoints_generator}

    def data_generator_sources(self):
        """
        Returns a dict that defines the connection between datasources and datagenerator parameters for their get() function.
        :return: A dict.
        """
        return {'image': {'image': 'image_datasource'},
                'landmarks_heatmap': {'landmarks': 'landmarks_datasource'},
                'landmarks_2Dpoints': {'landmarks': 'landmarks_datasource'}}

    def dataset_train(self):
        """
        Returns the training dataset. Random augmentation is performed.
        :return: The training dataset.
        """
        data_sources = self.data_sources(True)
        data_generator_sources = self.data_generator_sources()
        data_generators = self.data_generators(self.intensity_postprocessing_augmented)
        image_transformation = self.spatial_transformation_augmented()
        iterator = IdListIterator(self.cv_file_path,
                                  random=True,
                                  keys=['image_id'])
        dataset = ReferenceTransformationDataset(dim=self.dim,
                                                 reference_datasource_keys={'image': 'image_datasource'},
                                                 reference_transformation=image_transformation,
                                                 datasources=data_sources,
                                                 data_generators=data_generators,
                                                 data_generator_sources=data_generator_sources,
                                                 iterator=iterator,
                                                 debug_image_folder='debug_train_legacy' if self.save_debug_images else None)
        return dataset

    def dataset_val(self):
        """
        Returns the validation dataset. No random augmentation is performed.
        :return: The validation dataset.
        """
        data_sources = self.data_sources(False, self.input_image_ext)
        data_generator_sources = self.data_generator_sources()
        data_generators = self.data_generators(self.intensity_postprocessing)
        image_transformation = spatial_transformation(self.image_size)
        iterator = IdListIterator(self.cv_file_path,
                                  random=False,
                                  keys=['image_id'])
        dataset = ReferenceTransformationDataset(dim=self.dim,
                                                 reference_datasource_keys={'image': 'image_datasource'},
                                                 reference_transformation=image_transformation,
                                                 datasources=data_sources,
                                                 data_generators=data_generators,
                                                 data_generator_sources=data_generator_sources,
                                                 iterator=iterator,
                                                 debug_image_folder='debug_val_legacy' if self.save_debug_images else None)
        return dataset

    def get_data(self, xray_file_name_no_ext: str):
        dic = {'image_id': xray_file_name_no_ext}
        i = self.dataset.get(dic)
        xray_image = i['generators']['image']
        xray_target_landmarks = i['datasources']['landmarks_datasource']
        xray_target_landmarks_heatmap_space = i['generators']['landmarks_2Dpoints']
        reference_image = i['datasources']['image_datasource']
        xray_landmarks_points_np = np.empty(shape=(self.num_landmarks, self.dim + 1))
        for i in range(len(xray_target_landmarks)):
            cur_landmark_np = xray_landmarks_points_np[i]
            cur_landmark = xray_target_landmarks[i]
            cur_landmark_np[0] = float(cur_landmark.is_valid)
            cur_landmark_np[1:] = cur_landmark.coords[0], cur_landmark.coords[1]
        return xray_image, xray_landmarks_points_np, xray_target_landmarks_heatmap_space, reference_image.GetSize()

    def get_reference_data(self):
        dataset_entry = self.dataset.get_next()
        datasources = dataset_entry['datasources']
        transformations = dataset_entry['transformations']
        reference_image = datasources['image_datasource']
        reference_transform = transformations['image']
        return reference_image, reference_transform
