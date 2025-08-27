# deepfake-detection-system
Deepfake detection system using video as the primary modality with audio and spatio-temporal features for improved accuracy.

## 📌 Overview
This project presents a **video-primary deepfake detection framework** that integrates spatio-temporal video features with complementary audio features (MFCC + LSTM).  
The system achieves high accuracy in distinguishing **REAL** vs **FAKE** videos by prioritizing video modality while using audio as reinforcement.  

Key highlights:
- **R(2+1)D CNN (ResNet-18 backbone)** for spatio-temporal video encoding  
- **LSTM** for audio sequence modeling  
- **Late fusion** of modalities before classification  
- **FastAPI backend** and **React + Tailwind frontend** (optional)  
- Evaluation metrics: **Accuracy, Precision, Recall, F1-score, AUC-ROC**

---

## 📂 Datasets Used
This project was trained and evaluated using the following publicly available datasets:

- **[FakeAVCeleb](https://github.com/DASH-Lab/FakeAVCeleb)**  
  Multimodal deepfake dataset containing manipulated videos with both audio and visual modifications.
  @misc{khalid2021fakeavceleb,
    title        = {FakeAVCeleb: A Novel Audio-Video Multimodal Deepfake Dataset},
    author       = {Hasam Khalid and Shahroz Tariq and Simon S. Woo},
    year         = {2021},
    eprint       = {2108.05080},
    archivePrefix= {arXiv},
    primaryClass = {cs.CV}
  }

- **[CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D)**  
  Crowd-sourced emotional multimodal actors dataset used for audio augmentation.
  @article{cao2014crema,
  title   = {CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset},
  author  = {Haoqi Cao and Devang Kulkarni and Matthew J. Wilkins and Jose C. G. Neto and Shrikanth Narayanan and Maja J. Matari{\'c}},
  journal = {IEEE Transactions on Affective Computing},
  year    = {2014},
  doi     = {10.1109/TAFFC.2014.2336244}
}

- **[Meta's Casual Conversations Dataset](https://ai.meta.com/datasets/casual-conversations/)**  
  Dataset containing real conversational videos, used for bias testing and evaluation.
  @inproceedings{hazirbas2021casual,
  title     = {Casual Conversations: A Dataset for Measuring Fairness in AI},
  author    = {Caner Hazirbas and Tajana Radevski and David Ma and Charlie Hewitt and Mihaela Radu and Thomas Leung and Zeynep Akata},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2021},
  pages     = {3574-3584}
}

⚠️ *Note:* Due to dataset licensing and size, raw data is **not included** in this repository.  
To reproduce results, please download datasets from the official sources above.

---

## 📊 Results
- **Accuracy**: 96.69%  
- **AUC-ROC**: 0.9909  
- **Classification Report**:
  - REAL → Precision: 0.97, Recall: 0.98, F1-score: 0.97  
  - FAKE → Precision: 0.97, Recall: 0.95, F1-score: 0.96  

![ROC Curve](docs/Roccurve.png)  
![Confusion Matrix](docs/Confusion_matrix.png)

---

## ⚙️ Installation & Usage
Clone the repository:
git clone https://github.com/<your-username>/deepfake-detection-system.git
cd deepfake-detection-system

Backend (FastAPI)
cd backend
pip install -r requirements.txt
uvicorn main:app --reload

Frontend (React + Vite + Tailwind)
cd frontend
npm install
npm run dev

--

## 📜 License
This project is licensed under the
Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).
You are free to use, share, and adapt this work for non-commercial purposes only, with proper attribution.

--

## 📖 How to Cite
If you use this work in your research, please cite as:

@misc{karunarathna2025deepfake,
  author       = {Rasandi Karunarathna},
  title        = {Video-Centric Deepfake Detection: Leveraging Audio and Spatio-Temporal Features for Improved Accuracy},
  year         = {2025},
  howpublished = {\url{[https://github.com/Rasa-20/deepfake-detection-system.git}},
}
