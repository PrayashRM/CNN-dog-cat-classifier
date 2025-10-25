<!DOCTYPE html>
<html lang="en">

<body>

<h1 align="center">🐾 Animal Faces Classifier – CNN from Scratch (PyTorch)</h1>

<p align="center">
  <b>Deep Learning | CNN | Image Classification</b><br>
  Built <b>from scratch</b> using <b>PyTorch</b> — classifies Dogs, Cats, and Wild Animals with <b>96.775% accuracy</b> 🧠🔥
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-orange?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/Status-Completed-success" alt="Status">
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" alt="License">
</p>

<hr>

<h2>🚀 Overview</h2>

<p>
A deep learning model built from scratch using <b>vanilla CNN architecture</b> in <b>PyTorch</b> to classify animal images — Dogs, Cats, and Wild Animals — from the <a href="https://www.kaggle.com/datasets/andrewmvd/animal-faces/data">Kaggle Animal Faces Dataset</a>.<br><br>
This project demonstrates core <b>computer vision</b> and <b>model-building</b> skills without relying on any pre-built models or transfer learning techniques.
</p>

<hr>

<h2>🧠 Key Features</h2>

<ul>
  <li>⚙️ Fully implemented <b>from scratch</b> using PyTorch</li>
  <li>🧩 Custom CNN architecture (no pretrained models)</li>
  <li>🎛️ Data augmentation and regularization for robustness</li>
  <li>📈 Achieved <b>96.775% test accuracy</b></li>
  <li>🧱 Modular design for easy experimentation</li>
</ul>

<hr>

<h2>🧩 Dataset</h2>

<ul>
  <li><b>Dataset Used:</b> <a href="https://www.kaggle.com/datasets/andrewmvd/animal-faces/data">Animal Faces Dataset (Kaggle)</a></li>
  <li><b>Classes:</b> Dog 🐶 | Cat 🐱 | Wild Animals 🦁</li>
  <li><b>Data Split:</b>
    <ul>
      <li>Training Set – 70%</li>
      <li>Validation Set – 15%</li>
      <li>Test Set – 15%</li>
    </ul>
  </li>
</ul>

<hr>

<h2>🖼️ Data Preprocessing & Augmentation</h2>

<p>Each image is resized and augmented to improve generalization.</p>

<pre><code>transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor()
])
</code></pre>

<hr>

<h2>🏗️ Model Architecture</h2>

<pre><code>class MyCNNmodelNet(nn.Module):
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
</code></pre>

<hr>

<h2>⚙️ Training Details</h2>

<ul>
  <li>🎯 <b>Loss Function:</b> CrossEntropyLoss()</li>
  <li>⚡ <b>Optimizer:</b> Adam</li>
  <li>💻 <b>Device:</b> GPU (CUDA if available)</li>
  <li>🧩 <b>Framework:</b> PyTorch</li>
</ul>

<pre><code>criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)
</code></pre>

<hr>

<h2>📊 Model Evaluation</h2>

<p>Evaluation performed on the test set to measure average loss and accuracy.</p>

<pre><code>model.eval()
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
</code></pre>

<p align="center"><b>✅ Final Accuracy: 96.775%</b></p>

<hr>

<h2>🧰 Technologies Used</h2>

<p align="center">
  🐍 Python 3.x • 🔥 PyTorch • 🖼️ Torchvision • 📊 Scikit-learn • 📈 Matplotlib • 🧮 NumPy • 🧱 Pandas • 🖼️ Pillow
</p>

<hr>

<h2>🧪 Environment Setup</h2>

<pre><code># Clone the repository
git clone https://github.com/prayashmohanty/animal-faces-classifier.git
cd animal-faces-classifier

# Install dependencies
pip install torch torchvision scikit-learn matplotlib pandas pillow

# Run the training script
python train.py
</code></pre>

<hr>

<h2>🏁 Results</h2>

<table align="center">
  <tr>
    <th>Metric</th>
    <th>Value</th>
  </tr>
  <tr>
    <td>Accuracy</td>
    <td>96.775%</td>
  </tr>
  <tr>
    <td>Loss</td>
    <td>Low and stable across epochs</td>
  </tr>
  <tr>
    <td>Model Type</td>
    <td>Custom CNN</td>
  </tr>
  <tr>
    <td>Dataset</td>
    <td>Animal Faces (Kaggle)</td>
  </tr>
</table>

<hr>

<h2>🔮 Future Scope</h2>

<ul>
  <li>🚀 Experiment with Transfer Learning (ResNet, VGG16)</li>
  <li>🌐 Build a Flask/Streamlit web interface for live classification</li>
  <li>☁️ Deploy the model on cloud platforms (AWS, GCP, Hugging Face)</li>
  <li>🐾 Add more animal classes for broader generalization</li>
</ul>

<hr>

<h2>👨‍💻 Author</h2>

<p align="center">
  <b>Prayash Ranjan Mohanty</b><br>
  B.Tech in Computer Science (AI & ML)<br>
  Kalinga Institute of Industrial Technology, Bhubaneswar<br>
  📧 <a href="mailto:prayashranjanmohanty11@gmail.com">prayashranjanmohanty11@gmail.com</a>
</p>

<p align="center">
  <a href="https://github.com/prayashmohanty">
    <img src="https://img.shields.io/badge/GitHub-PrayashRanjanMohanty-black?logo=github" alt="GitHub">
  </a>
</p>

<hr>

<h2>🪪 License</h2>

<p align="center">
  This project is released under the <b>MIT License</b> — free for personal and academic use. 🧾
</p>

</body>
</html>
