#  Weapon Detection System using YOLOv8 on Edge (Jetson Nano)

---

## 1.  Project Title

**Weapon Detection System using YOLOv8 with Edge Computing (Jetson Nano)**

---

## 2.  Problem Statement

The increasing need for security in public spaces such as schools, exam halls, and transport hubs requires automated systems to detect dangerous objects like weapons in real time.

Manual monitoring is:

* Time-consuming
* Error-prone
* Not scalable

This project aims to build an **AI-powered weapon detection system** capable of detecting weapons in real time using computer vision.

---

## 3.  Role of Edge Computing

This system leverages **edge computing using Jetson Nano** to process data locally instead of relying on cloud servers.

###  Components running on Jetson Nano:

* YOLOv8 model inference
* Video frame processing (OpenCV)
* Real-time detection output

###  Why Edge instead of Cloud?

*  **Low latency** → instant detection
*  **Offline capability** → works without internet
*  **Better privacy** → no data sent to cloud
*  **Efficient processing** → optimized for real-time use

---

## 4.  Methodology / Approach

###  System Pipeline:

**Input → Preprocessing → Model → Output**

###  Steps:

1. **Input**

   * Video stream / webcam / image input

2. **Preprocessing**

   * Frame resizing
   * Normalization
   * Noise reduction

3. **Model**

   * YOLOv8 detects weapons using bounding boxes

4. **Output**

   * Detected objects displayed with labels
   * Confidence score shown

---

## 5.  Model Details

* Model: **YOLOv8 (Ultralytics)**
* Type: CNN-based Object Detection Model
* Framework: PyTorch
* Input Size: 640x640 images
* Output: Bounding boxes + class labels

###  Optimization (Optional)

* Can be optimized using **TensorRT** for Jetson Nano to improve FPS

---

## 6.  Training Details

* Dataset: Custom weapon detection dataset
* Classes: Gun, Knife (example)
* Training performed using YOLOv8 framework

###  Training Process:

* Data annotation
* Model training using epochs
* Validation after each epoch

###  Metrics:

* Loss vs Epoch
* Accuracy vs Epoch

(Note: Graphs can be added in future)

---

## 7.  Results / Output

###  Output:

* Real-time detection of weapons
* Bounding boxes around detected objects

###  Performance:

* System works in real-time
* FPS depends on hardware

###  Comparison:

| Platform      | Performance             |
| ------------- | ----------------------- |
| Normal Laptop | Higher FPS              |
| Jetson Nano   | Optimized but lower FPS |

---

## 8.  Setup Instructions

###  Clone Repository:

```bash
git clone https://github.com/nsingh3be24-droid/WEAPON-DETECTION-SYSTEM.git
cd WEAPON-DETECTION-SYSTEM
```

###  Install Dependencies:

```bash
pip install -r requirements.txt
```

###  Run Project:

```bash
python main.py
```

---

##  Notes

* Model weights (.pt files) are not uploaded due to GitHub size limits
* Dataset is also excluded; sample data can be added

---

##  Author

Navraj Singh

---
