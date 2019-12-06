import numpy as np
from sklearn.model_selection import train_test_split

data = np.load('/opt/ml/processing/input/data.npy')

train, test = train_test_split(data, test_size=0.2)
train, validation = train_test_split(train, test_size=0.5)

try:
    os.makedirs('/opt/ml/processing/output/train')
    os.makedirs('/opt/ml/processing/output/validation')
    os.makedirs('/opt/ml/processing/output/test')
except:
    pass

np.save('/opt/ml/processing/output/train/train.npy', train)
np.save('/opt/ml/processing/output/validation/validation.npy', validation)
np.save('/opt/ml/processing/output/test/test.npy', test)
