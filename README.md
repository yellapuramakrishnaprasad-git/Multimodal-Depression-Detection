# Multimodal Depression Detection using Deep Learning & Machine Learning

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-1.14.0-orange.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-2.3.1-red.svg)](https://keras.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-0.22.2-green.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)

An intelligent, multi-modal depression detection application built with **Python**, **Deep Learning (CNN)**, **Random Forest**, and **Tkinter GUI**. This application detects indicators of depression across three distinct input modalities: **Text comments**, **Facial Expression Images**, and **Speech Audio Signals**.

---

## 🌟 Overview & Features

Depression is a prevalent mental health disorder that manifests through multiple communication channels—verbal expression, facial affect, and acoustic characteristics. This system provides a unified framework for multi-modal analysis:

- 📝 **Text Modality**: Processes user comments and text data using **TF-IDF (Term Frequency-Inverse Document Frequency)** feature extraction and a **Random Forest Classifier** to identify depression indicators.
- 🖼️ **Facial Image Modality**: Utilizes a **Convolutional Neural Network (CNN)** trained on facial expressions to detect mood and depression cues.
- 🎙️ **Speech Audio Modality**: Extracts acoustic features (**MFCC**, **Chroma STFT**, and **Mel Spectrogram**) via **Librosa** and processes them through a **CNN Classifier** to analyze voice tone and emotion.
- 📊 **Performance Analytics**: Generates real-time comparison graphs showing **Accuracy**, **Precision**, **Recall**, and **F1-Score** across all three modalities.
- 🖥️ **User-Friendly GUI**: A desktop interface built with **Tkinter** allowing seamless dataset loading, model training, performance evaluation, and live inference.

---

## 📁 Repository Structure

```gfm
DepressionDetection/
├── Dataset/
│   ├── Images/                 # Facial expression dataset categorized by emotions
│   ├── Speech/                 # Speech audio files categorized by actors/emotions
│   └── Suicide_Detection.csv   # Text dataset containing labeled comments
├── model/                      # Pretrained weights, JSON model architecture & feature cache
│   ├── cnnmodel.json
│   ├── cnnmodel_weights.h5
│   ├── speechmodel.json
│   ├── speech_weights.h5
│   ├── tfidf.pckl
│   └── *.npy                   # Preprocessed dataset numpy arrays
├── model1/                     # Alternative/backup model weight checkpoints
├── testImages/                 # Sample facial images for live prediction
├── testSpeech/                 # Sample audio WAV files for live prediction
├── testText/                   # Sample CSV text comments for live prediction
├── Main.py                     # Primary GUI Application & Model Training Pipeline
├── requirements.txt            # Project dependencies and versions
├── run.bat                     # Windows batch script to launch application
├── SCREENS.docx                # Screenshots of GUI and execution results
└── README.md                   # Project documentation
```

---

## 🛠️ Modality Architecture & Algorithms

### 1. Text Analysis (Random Forest)
- **Preprocessing**: Removal of English stopwords and short tokens using `NLTK`.
- **Feature Extraction**: `TfidfVectorizer` transforms text comments into high-dimensional numerical vectors.
- **Classification**: `RandomForestClassifier` trained to distinguish between *Depressed* and *Non-Depressed* statements.

### 2. Facial Expression Analysis (2D CNN)
- **Input**: Resized `32x32x3` RGB facial image tensors normalized to `[0, 1]`.
- **Architecture**:
  - Convolutional Layer (32 filters, 3x3 kernel, ReLU activation) -> MaxPooling (2x2)
  - Convolutional Layer (32 filters, 3x3 kernel, ReLU activation) -> MaxPooling (2x2)
  - Flatten Layer -> Dense Layer (256 units, ReLU) -> Dense Output (Softmax)
- **Output**: Categorizes facial state into emotional categories (*Happy*, *Neutral*, *Sad*, *Depressed*, etc.).

### 3. Speech Audio Analysis (Audio CNN)
- **Feature Extraction**: Extracts 180 total acoustic features per audio signal using `Librosa` & `SoundFile`:
  - **MFCC** (Mel-Frequency Cepstral Coefficients, 40 features)
  - **Chroma** (Short-Time Fourier Transform)
  - **Mel Spectrogram**
- **Classification**: Reshaped feature maps passed through a Convolutional Neural Network with categorical cross-entropy loss.

---

## ⚙️ Installation & Setup Instructions

### Prerequisites
- **Python 3.7+** (Recommended: Python 3.7 or 3.8 for compatibility with TensorFlow 1.14 / Keras 2.3)
- **Git** & **Git LFS** (for downloading large model weights and datasets)

### Step 1: Clone the Repository
```bash
git clone https://github.com/<YOUR_USERNAME>/<YOUR_REPO_NAME>.git
cd <YOUR_REPO_NAME>
```

### Step 2: Install Git LFS (Large File Storage)
Since the repository contains trained model weights and large datasets, pull Git LFS tracked files:
```bash
git lfs install
git lfs pull
```

### Step 3: Install Required Dependencies
It is recommended to use a virtual environment:
```bash
# Create and activate virtual environment (optional)
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Running the Application

### Option A: Via Command Line
```bash
python Main.py
```

### Option B: Via Windows Batch File
Double-click `run.bat` or execute in PowerShell:
```cmd
.\run.bat
```

---

## 💻 How to Use the GUI Application

1. **Upload Depression Dataset**: Click to browse and load the dataset folder containing Text, Image, and Speech data.
2. **Preprocess Dataset**: Preprocesses images (resizing & normalization), audio features, and text TF-IDF matrices.
3. **Train Facial Depression CNN**: Trains/loads the facial expression CNN model and outputs training metrics.
4. **Train Speech Depression CNN**: Trains/loads the audio feature CNN model and displays accuracy/F1-score.
5. **Train Text Random Forest**: Fits the Random Forest classifier on text TF-IDF features.
6. **Accuracy Comparison Graph**: Displays a comparative bar graph of Accuracy, Precision, Recall, and F1-Score across all modalities.
7. **Predict Modalities**:
   - **Predict Facial Depression**: Select an image from `testImages/` to view real-time facial sentiment overlay.
   - **Predict Speech Depression**: Select a WAV audio file from `testSpeech/` to classify acoustic tone.
   - **Predict Text Depression**: Select a test CSV file from `testText/` to evaluate text comments.

---

## 📊 Performance Metrics Comparison

| Modality / Model | Algorithm | Primary Features | Output Classes |
| :--- | :--- | :--- | :--- |
| **Text** | Random Forest | TF-IDF Vectorization | Depressed / Non-Depressed |
| **Facial Image** | Convolutional Neural Network (CNN) | 32x32 RGB Image Tensors | Happy, Neutral, Sad, Depressed |
| **Speech Audio** | Convolutional Neural Network (CNN) | MFCC, Chroma, Mel Spectrogram | Depressed / Non-Depressed |

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 🤝 Contributing & Acknowledgments
Contributions, issues, and feature requests are welcome! Feel free to check the issues page or submit pull requests.
