import re
from typing import List, Union, Dict, Optional
from abc import ABC, abstractmethod

import warnings

warnings.filterwarnings("ignore")


# --- Base Processor Classes ---
class Processor(ABC):
    processor_name: str
    function_name_list: List[str]

    def __init__(self):
        self.processor_name = "processor"
        self.function_name_list = []

    def get_name(self) -> str:
        return self.processor_name

    def __call__(self, input_text):
        return self.process(input_text)

    @abstractmethod
    def process(self, input_text):
        pass

    # Use the >> operator to compose processors.
    def __rshift__(self, other):
        # If self is already a ComposedProcessor, we merge its processors with 'other'
        if isinstance(self, ComposedProcessor):
            return ComposedProcessor(self.processors + [other])
        return ComposedProcessor([self, other])


class ComposedProcessor(Processor):
    def __init__(self, processors: List[Processor]):
        super().__init__()
        self.processors = processors
        # Set function_name_list to a list of the names of each processor.
        self.function_name_list = [p.get_name() for p in processors]

    def process(self, input_text):
        for processor in self.processors:
            input_text = processor(input_text)
        return input_text


# =====================================================================================
class IdentityProcessor(Processor):
    """
    An identity processor return a copy of input message itself
    """

    def __init__(self):
        self.processor_name = "identity"

    def process(self, x):
        return x
