import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from google.colab import files
uploaded = files.upload()

dataset = pd.read_csv("Dataset.csv")
print(dataset.shape)
print(dataset.head(5))

#Upper Confidence Bound

import math
observations = 10000
