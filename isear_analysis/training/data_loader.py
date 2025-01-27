import pandas as pd

class DataLoader:
  def __init__(self, dataset_path):
    self.load_dataset(dataset_path)
    pass

  def load_dataset(self, dataset_path):
    try:
      self.dataset = pd.read_csv(dataset_path)
    except FileNotFoundError:
      print(f'Data not found with path: {dataset_path}')
