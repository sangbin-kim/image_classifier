import functions
from torch import nn
from torch import optim

from torchvision import models

# 1. 이미지 파일 디렉로티 지정
data_dir = r'D:\#.Secure Work Folder\1. Data\1. CMI\1. FLD\train_wo_ok'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# 2. Data Loading and Transforms

train_loader, validate_loader, test_loader, training_dataset, validation_dataset, testing_dataset = functions.loader(train_dir, valid_dir, test_dir)


# 3. Label Mapping
import json

with open('defect_type.json', 'r') as f:
    defect_type = json.load(f)

print(len(defect_type))
print(defect_type)

# 4. model selection

model = models.vgg11()

# 5. Build Custom Classifier

from collections import OrderedDict

classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(25088, 5000)),
                                        ('relu', nn.ReLU()),
                                        ('drop', nn.Dropout(p=0.5)),
                                        ('fc2', nn.Linear(5000, 3)),
                                        ('output', nn.LogSoftmax(dim=1))]))
model.classifier = classifier

# 6. Loss function and gradient descent

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 7. model training

model = functions.train_classifier(model, train_loader, validate_loader, optimizer, criterion)

# 8. test accuracy

functions.test_accuracy(model, test_loader)

# 9. model save

model_save_path = r"C:/Users/LG/Desktop/ksb/3. CODE/DeepLearning/"
filename = 'test.pth'
functions.save_checkpoint(model, training_dataset, model_save_path, filename)