import torch
import torchvision.transforms as tf
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report
from model import CNN_model_3
from helpers import DeviceDataLoader, get_device

# Step 1: Load model
device = get_device()
model = CNN_model_3(opt_fun=torch.optim.Adam, lr=0.001)
model.load_state_dict(torch.load("cnn_model_trained.pt", map_location=device))
model.eval()
model.to(device)

# Step 2: Load test dataset
transformations = tf.Compose([tf.Resize([64,64]), tf.ToTensor()])
testset = ImageFolder('new_data/test', transform=transformations)
classes = testset.classes
test_dl = DeviceDataLoader(DataLoader(testset, batch_size=64), device)

# Step 3: Run model and collect predictions
y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in test_dl:
        outputs = model(images)
        _, preds = torch.max(outputs, dim=1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# Step 4: Print classification report
print("F1 Score Report (per language):\n")
print(classification_report(y_true, y_pred, target_names=classes))
