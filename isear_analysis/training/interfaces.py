from dataclasses import dataclass

@dataclass
class DatasetInformation:
    """Class to save attribute of Dataset Information needed for data loader
    """
    dataset_path: str
    tokenizer_name: str
    text_column_name: str
    label_column_name: str