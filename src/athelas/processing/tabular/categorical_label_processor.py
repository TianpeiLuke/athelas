# processor/categorical_label_processor.py
from typing import List
from .processors import Processor

class CategoricalLabelProcessor(Processor):
    def __init__(self, initial_categories: List[str] = None, update_on_new: bool = True, unknown_label: int = -1):
        """
        Args:
            initial_categories (List[str], optional): Initial list of categories.
            update_on_new (bool): If True, add new categories to the mapping as they are encountered.
            unknown_label (int): Label to assign if update_on_new is False and a new category is encountered.
        """
        super().__init__()
        self.processor_name = "categorical_label_processor"
        if initial_categories is None:
            self.category_to_label = {}
            self.next_label = 0
        else:
            self.category_to_label = {cat: idx for idx, cat in enumerate(initial_categories)}
            self.next_label = len(initial_categories)
        self.update_on_new = update_on_new
        self.unknown_label = unknown_label
    
    def process(self, input_text: str) -> int:
        # Transform category string into a numeric label.
        if input_text in self.category_to_label:
            return self.category_to_label[input_text]
        else:
            if self.update_on_new:
                self.category_to_label[input_text] = self.next_label
                self.next_label += 1
                return self.category_to_label[input_text]
            else:
                return self.unknown_label
