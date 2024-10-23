from abc import ABC, abstractmethod


class MedicalDataAugmentationBase(ABC):
    @abstractmethod
    def get_data(self, xray_file_name: str):
        pass

