🐾 Animal Faces Classifier – CNN from Scratch (PyTorch)

A deep learning model built from scratch using vanilla CNN architecture in PyTorch to classify animal images (dogs, cats, and wild animals) from the Kaggle Animal Faces Dataset
.
This project demonstrates core computer vision and model-building skills without relying on any pre-built models or transfer learning techniques.

🚀 Overview

The model classifies images of animals into three categories — Dogs, Cats, and Wild Animals — using a custom-built convolutional neural network (CNN).
It achieves an impressive accuracy of 96.775%, showcasing effective training, data preprocessing, and model design.

🧠 Key Features

Fully implemented from scratch using PyTorch

Custom CNN architecture (no pretrained models)

Data augmentation and regularization for robustness

Achieved 96.775% test accuracy

Modular code design for easy experimentation

🧩 Dataset

Dataset Used: Animal Faces Dataset – Kaggle

Classes: Dog, Cat, Wild Animals

Split:

Training Set: 70%

Validation Set: 15%

Test Set: 15%

🖼️ Data Preprocessing & Augmentation

Each image is resized and augmented to improve generalization.

transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor()
])

🏗️ Model Architecture

The CNN model is implemented from scratch using PyTorch’s nn.Module.

class MyCNNmodelNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pooling = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(128, 128)
        self.output = nn.Linear(128, len(data['label'].unique()))

    def forward(self, x):
        x = self.pooling(self.relu(self.conv1(x)))
        x = self.pooling(self.relu(self.conv2(x)))
        x = self.pooling(self.relu(self.conv3(x)))
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.relu(self.linear(x))
        x = self.output(x)
        return x

⚙️ Training Details

Loss Function: CrossEntropyLoss()

Optimizer: Adam

Learning Device: GPU (CUDA if available)

Framework: PyTorch

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

📊 Model Evaluation

Evaluation was performed on the test set to measure accuracy and average loss.

model.eval()
correct, total, total_epoch_loss = 0, 0, 0.0

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        prediction = model(x_batch)
        loss = criterion(prediction, y_batch)
        total_epoch_loss += loss.item()

        predicted_class = torch.argmax(prediction, dim=1)
        correct += (predicted_class == y_batch).sum().item()
        total += y_batch.size(0)

avg_loss = total_epoch_loss / len(test_loader)
accuracy = (correct / total) * 100
print(f'Avg Loss: {avg_loss:.4f} | Accuracy: {accuracy:.3f}%')


📈 Final Accuracy: 96.775%

🧰 Technologies Used

Python 3.x

PyTorch

Torchvision

Scikit-learn

Matplotlib

Pandas

NumPy

Pillow

🧪 Environment Setup

Clone the repository:

git clone https://github.com/yourusername/animal-faces-classifier.git
cd animal-faces-classifier


Install dependencies:

pip install torch torchvision scikit-learn matplotlib pandas pillow


Mount or place dataset in the project directory (if using Google Colab, mount Google Drive).

Run the training script:

python train.py

🏁 Results
Metric	Value
Accuracy	96.775%
Loss	Low and stable across epochs
Model Type	Custom CNN
Dataset	Animal Faces (Kaggle)
🔮 Future Scope

Experiment with Transfer Learning (e.g., ResNet, VGG16)

Build Flask/Streamlit web interface for live classification

Deploy model on cloud platforms (AWS, GCP, Hugging Face)

Add more animal classes for broader generalization

👨‍💻 Author

Prayash Ranjan Mohanty
B.Tech in Computer Science (AI & ML)
Kalinga Institute of Industrial Technology, Bhubaneswar
📧 prayashranjanmohanty11@gmail.com

🪪 License

This project is released under the MIT License — free for personal and academic use.
