import numpy as np
from typing import List, Union, Optional, Dict


from ..processors import Processor


class MultiClassLabelProcessor(Processor):
    """
    Processes multi-class labels into a format suitable for machine learning models.

    This processor handles encoding categorical labels into numerical arrays
    and optionally provides one-hot encoding using numpy.

    Args:
        label_list (Optional[List[str]]): A list of unique label strings. If provided,
                                     the processor will learn this mapping; otherwise,
                                     it will learn the mapping from the data it processes.
        one_hot (bool): If True, output one-hot encoded labels.
        strict (bool): If True, raise error for unknown labels when label_list is provided.
    """

    def __init__(
        self,
        label_list: Optional[List[str]] = None,
        one_hot: bool = False,
        strict: bool = False,
    ):
        super().__init__()
        self.processor_name = "multiclass_label_processor"
        self.label_to_id: Dict[str, int] = {}
        self.id_to_label: List[str] = []
        self.one_hot = one_hot
        self.strict = strict

        if label_list:
            self.label_to_id = {
                str(label): i for i, label in enumerate(label_list)
            }  # {label: i for i, label in enumerate(label_list)}
            self.id_to_label = list(map(str, label_list))  # label_list

    def process(self, labels: Union[str, List[str]]) -> np.ndarray:
        """
        Encodes the input labels.

        Args:
            labels (Union[str, List[str]]): A single label or a list of labels.

        Returns:
            np.ndarray: Encoded labels as a numpy array.
        """

        if isinstance(labels, (str, int, float)):
            labels = [labels]  # Wrap scalar in list

        encoded_labels = []
        for label in labels:
            label = str(label)  # Normalize all labels to string
            if label not in self.label_to_id:
                if self.strict:
                    raise ValueError(f"Label '{label}' not found in known label list.")
                self.label_to_id[label] = len(self.label_to_id)
                self.id_to_label.append(label)
            encoded_labels.append(self.label_to_id[label])

        encoded_array = np.array(encoded_labels, dtype=np.int64)

        if self.one_hot:
            # Create one-hot encoding using numpy
            num_classes = len(self.id_to_label)
            one_hot_labels = np.eye(num_classes)[encoded_array].astype(np.float32)
            return one_hot_labels
        else:
            return encoded_array
