
import torch
from torch import nn
from torchvision import datasets, transforms, models
from collections import OrderedDict


def loader(train_dir,valid_dir,  test_dir):

    # Define transforms for the training, validation, and testing sets
    training_transforms = transforms.Compose([transforms.Resize(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])

    testing_transforms = transforms.Compose([transforms.Resize(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406],
                                                                  [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    training_dataset = datasets.ImageFolder(train_dir, transform=training_transforms)
    validation_dataset = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    testing_dataset = datasets.ImageFolder(test_dir, transform=testing_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=8, shuffle=True)
    validate_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=4)
    test_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=4)

    return train_loader, validate_loader, test_loader, training_dataset, validation_dataset, testing_dataset


# Function for the validation pass
def validation(model, validateloader, criterion):
    val_loss = 0
    accuracy = 0

    for images, labels in iter(validateloader):
        images, labels = images.to('cuda'), labels.to('cuda')

        output = model.forward(images)
        val_loss += criterion(output, labels).item()

        probabilities = torch.exp(output)

        equality = (labels.data == probabilities.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return val_loss, accuracy


# Train the classifier

def train_classifier(model, train_loader, validate_loader, optimizer, criterion):
    epochs = 15
    steps = 0
    print_every = 40

    model.to('cuda')

    for e in range(epochs):

        model.train()

        running_loss = 0

        for images, labels in iter(train_loader):

            steps += 1
            if torch.cuda.is_available():
                images, labels = images.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()

            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()

                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    validation_loss, accuracy = validation(model, validate_loader, criterion)

                print("Epoch: {}/{}.. ".format(e + 1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss / print_every),
                      "Validation Loss: {:.3f}.. ".format(validation_loss / len(validate_loader)),
                      "Validation Accuracy: {:.3f}".format(accuracy / len(validate_loader)))

                running_loss = 0
                model.train()
    return model


def test_accuracy(model, test_loader):
    # Do validation on the test set
    model.eval()
    model.to('cuda')

    with torch.no_grad():
        accuracy = 0

        for images, labels in iter(test_loader):
            images, labels = images.to('cuda'), labels.to('cuda')

            output = model.forward(images)

            probabilities = torch.exp(output)

            equality = (labels.data == probabilities.max(dim=1)[1])

            accuracy += equality.type(torch.FloatTensor).mean()

        print("Test Accuracy: {}".format(accuracy / len(test_loader)))

def save_checkpoint(model, training_dataset, model_save_path, filename):

    model.class_to_idx = training_dataset.class_to_idx

    checkpoint = {'arch': "vgg11",
                  'class_to_idx': model.class_to_idx,
                  'model_state_dict': model.state_dict()
                 }

    torch.save(checkpoint, model_save_path + filename)
    print("Model saved")


def load_checkpoint(model_save_path, filename):
    checkpoint = torch.load(model_save_path + filename)

    if checkpoint['arch'] == 'vgg11':

        model = models.vgg11()

        for param in model.parameters():
            param.requires_grad = False
    else:
        print("Architecture not recognized.")

    model.class_to_idx = checkpoint['class_to_idx']

    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(25088, 5000)),
                                            ('relu', nn.ReLU()),
                                            ('drop', nn.Dropout(p=0.5)),
                                            ('fc2', nn.Linear(5000, 3)),
                                            ('output', nn.LogSoftmax(dim=1))]))

    model.classifier = classifier

    model.load_state_dict(checkpoint['model_state_dict'])

    return model


from PIL import Image


def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    #     # Process a PIL image for use in a PyTorch model

    pil_image = Image.open(image_path)
    pil_image = pil_image.convert('RGB')

    #     # Resize
    #     if pil_image.size[0] > pil_image.size[1]:
    #         pil_image.thumbnail((5000, 224))
    #     else:
    #         pil_image.thumbnail((224, 5000))

    #     # Crop
    #     left_margin = (pil_image.width-224)/2
    #     bottom_margin = (pil_image.height-224)/2
    #     right_margin = left_margin + 224
    #     top_margin = bottom_margin + 224

    #     pil_image = pil_image.crop((left_margin, bottom_margin, right_margin, top_margin))

    #     # Normalize
    #     np_image = np.array(pil_image)/255
    #     mean = np.array([0.485, 0.456, 0.406])
    #     std = np.array([0.229, 0.224, 0.225])
    #     print(np_image.shape)
    #     print(mean.shape)
    #     print(std.shape)
    #     np_image = (np_image - mean) / std
    img_transforms = transforms.Compose([transforms.Resize(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])
    img = img_transforms(pil_image)

    # PyTorch expects the color channel to be the first dimension but it's the third dimension in the PIL image and Numpy array
    # Color channel needs to be first; retain the order of the other two dimensions.
    #     np_image = np_image.transpose((2, 0, 1))

    return img


# Implement the code to predict the class from an image file

def predict(image_path, model, topk=3):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    image = process_image(image_path)

    #     # Convert image to PyTorch tensor first
    #     image = torch.from_numpy(image).type(torch.cuda.FloatTensor)
    #     #print(image.shape)
    #     #print(type(image))

    # Returns a new tensor with a dimension of size one inserted at the specified position.
    image = image.unsqueeze(0)

    output = model.forward(image)

    probabilities = torch.exp(output)

    # Probabilities and the indices of those probabilities corresponding to the classes
    top_probabilities, top_indices = probabilities.topk(topk)

    # Convert to lists
    top_probabilities = top_probabilities.detach().type(torch.FloatTensor).numpy().tolist()[0]
    top_indices = top_indices.detach().type(torch.FloatTensor).numpy().tolist()[0]

    # Convert topk_indices to the actual class labels using class_to_idx
    # Invert the dictionary so you get a mapping from index to class.

    idx_to_class = {value: key for key, value in model.class_to_idx.items()}
    # print(idx_to_class)

    top_classes = [idx_to_class[index] for index in top_indices]

    return top_probabilities, top_classes
