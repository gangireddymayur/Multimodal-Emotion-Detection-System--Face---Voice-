# 🎭 Multimodal Emotion Detection System (Face + Voice)

A Streamlit-based AI application that detects **human emotions from facial expressions and voice**, then performs **fusion** to give a final multimodal emotion result.

This system supports:
- **Webcam mode** → Face emotion only (real-time)
- **Upload video mode** → Face emotion + Voice emotion + Fusion decision

---

## 📌 Features

✅ Facial Emotion Recognition using **EfficientNet-B4 (PyTorch)**  
✅ Face detection using **OpenCV Haarcascade**  
✅ Voice Emotion Recognition using **MFCC + MLP model**  
✅ Multimodal Fusion (Face + Voice weighted decision)  
✅ Streamlit dashboard UI  
✅ Export emotion summary as CSV  
✅ GPU support (CUDA if available)

---

## 🧠 Emotions Supported

### Face Emotions (7 classes)
- Surprise
- Fear
- Disgust
- Happy
- Sad
- Angry
- Neutral

### Voice Emotions (4 classes)
- Angry
- Happy
- Sad
- Neutral

---

## 📂 Project Structure

```

multimodal-emotion-detection/
│
├── app.py
├── model.py
├── dataset.py
├── loss.py
├── utils.py
├── early_stopping.py
├── fusion.py
├── audio_utils.py
├── voice_emotion.py
├── train.py
│
├── best_efficientnet.pth


````

---

## ⚙️ Requirements

Install these Python packages:

- streamlit
- pandas
- numpy
- opencv-python
- torch
- torchvision
- pillow
- moviepy
- librosa
- scikit-learn

---

## ✅ Installation

### 1) Clone / Download project
```bash
git clone <your-repo-url>
cd multimodal-emotion-detection
````

### 2) Create virtual environment (Recommended)

```bash
python -m venv venv
```

Activate:

**Windows**

```bash
venv\Scripts\activate
```

**Linux/Mac**

```bash
source venv/bin/activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```


---

## ▶️ Run the Application

```bash
streamlit run app.py
```

App will open in browser automatically.

---

## 🎥 Modes

### ✅ 1) Webcam Mode (Face Only)

* Works in real-time using your webcam.
* Detects face emotion only.
* Shows:

  * Emotion label
  * Confidence score
  * Frames processed
  * Faces detected

### ✅ 2) Upload Video Mode (Face + Voice)

* Upload `.mp4 / .avi / .mov`
* System will:

  1. Process face emotions frame-by-frame
  2. Create emotion summary table
  3. Extract audio from video
  4. Detect voice emotion
  5. Fuse final multimodal emotion

---

## 🧠 Model Details

### ✅ Face Model

* Backbone: **EfficientNet-B4**
* Framework: PyTorch
* Input size: 224 × 224
* Output: 7-class emotion prediction
* Best weights saved as:

  ```
  best_efficientnet.pth
  ```

### ✅ Voice Model

* Features: **MFCC (40)**
* Model: Simple MLP
* Output: 4-class prediction

---

## 🔥 Fusion Logic

Fusion is done using rule + confidence weighting:

* If voice emotion is **Angry** and confidence ≥ 0.5 → final = Angry
* If both predictions match → final = same emotion
* Else → emotion with higher confidence wins

File:

```
fusion.py
```

---

## 📊 Output

In Upload mode, after processing:

* Summary table with emotion counts and avg confidence
* Bar chart
* Voice emotion result
* Final fused emotion
* Download results as CSV

---

## 📌 Notes / Important

* Make sure `best_efficientnet.pth` exists in project folder.
* Webcam mode requires camera access.
* Video upload uses temporary files:

  * `temp_video.mp4`
  * `temp_audio.wav`

---

## 🚀 Future Improvements (Optional)

* Use a pretrained voice emotion model (better accuracy)
* Use MediaPipe face detection for better detection
* Add real-time voice mode (mic emotion)
* Deploy to Streamlit Cloud / HuggingFace Spaces

---

## 👨‍💻 Author

Developed by: **(Gangireddy Mayur)**
Project: Multimodal Emotion Detection using Streamlit + PyTorch + OpenCV

---

