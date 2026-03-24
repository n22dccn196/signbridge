This is the comprehensive, technical-heavy **README.md** for your **SignBridge** repository, incorporating all data from your project documents, repository structure, and technical confirmations.

-----

# SignBridge: Real-time IoT Sign Language Translator

[](https://www.google.com/search?q=https://github.com/vinh-lth/SignBridge)
[](https://www.google.org/)
[](https://www.raspberrypi.org/)

**SignBridge** is a high-performance, privacy-preserving **Edge-Fog** hybrid system designed to bridge the communication gap between the Deaf and Hard-of-Hearing (DHH) community and the hearing public. Built for the **Google.org** initiative, it focuses on extreme low latency and data security in service environments (F\&B, Healthcare).

-----

## 🏗 System Architecture

The project implements a distributed inference pipeline to balance computational load and user privacy.

### 1\. Edge Layer (Raspberry Pi 4 + Camera)

  * **MediaPipe Integration:** Real-time extraction of 75 holistic keypoints (Hands, Pose, Face).
  * **Privacy-by-Design:** All video frames are processed in RAM and discarded immediately. No raw video is ever transmitted or stored.
  * **Communication:** Serialized coordinate data is streamed via **WebSockets** to the Fog Node.

### 2\. Fog Layer (Local PC / Server)

  * **Ensemble Engine:** A multi-head deep learning architecture that aggregates predictions from:
      * **TCN (Temporal Convolutional Network):** Captures local motion patterns.
      * **BiLSTM:** Models long-term temporal dependencies.
      * **Transformer:** Focuses on global spatial-temporal attention.
  * **Decision Logic:** Weighted probability averaging to achieve robust classification.

-----

## 📂 Repository Structure

```text
.
├── annotations/            # BOBSL metadata, pseudo-labels, and fingerspelling data
├── sign_language_web/      # React/Flask-based Dashboard for real-time visualization
├── do.py                   # Main pipeline: Training, Inference, and Data Management
├── temp.ipynb              # Prototyping, feature testing, and model evaluation
├── raw.csv                 # Dataset mapping and label indexing
├── hybrid_part1_aria2.txt  # Automated dataset retrieval (Aria2 scripts)
├── hybrid_part2_curl.bat   # Secondary data fetch scripts
└── bobsl_v1_4_...          # Raw and processed BOBSL dataset components
```

-----

## 🔬 Model & Performance

| Metric | Value |
| :--- | :--- |
| **Top-1 Accuracy** | **77.91%** |
| **Top-5 Accuracy** | **96.20%** |
| **End-to-End Latency** | **\< 500ms** |
| **Dataset** | BOBSL (British Sign Language) + Local Custom Sets |
| **Inference Mode** | Edge-Fog Distributed |

-----

## 🛠 Installation & Setup

### Prerequisites

  * **Edge:** Raspberry Pi 4 (4GB+ recommended), OV2640 or USB Camera.
  * **Fog:** Local PC with Python 3.9+ (NVIDIA GPU recommended for Ensemble inference).

### Step 1: Clone and Environment

```bash
git clone https://github.com/vinh-lth/SignBridge.git
cd SignBridge
pip install -r requirements.txt
```

### Step 2: Data Preparation

Use the provided scripts to download and extract the BOBSL dataset:

```bash
aria2c -i hybrid_part1_aria2.txt
python do.py --extract_annotations
```

### Step 3: Running the System

1.  **Start the Fog Node (PC):**
    ```bash
    python do.py --mode serve --port 8080
    ```
2.  **Start the Edge Node (Raspberry Pi):**
    ```bash
    python edge_stream.py --server_ip <YOUR_PC_IP>
    ```

-----

## 🤝 Team: Visionary Tech

  * **Lê Trang Hoàng Vinh** - Team Lead & ML Developer (Data, Training, Deployment).
 

-----

## ⚖️ Strategic Positioning (Vs. Competitors)

  * **Vs. Wearables (BrightSign):** SignBridge is **non-invasive**; no gloves required.
  * **Vs. Mobile Apps (HandTalk):** SignBridge offers **lower latency** and **higher privacy** via Edge processing instead of Cloud API.
  * **Vs. Enterprise (SignAll):** SignBridge is **cost-effective**, utilizing consumer-grade hardware (RPi 4) for mass deployment.

-----

## 📜 License

This project is developed for educational and social impact purposes. Please refer to `LICENSE` for further details.

-----

### **Final Conclusion & Next Steps**

I have synthesized your SWOT, Competitor Analysis, Hardware specs, and File structure into this README.

**Is there any specific "Hardware Connection Diagram" or "Mathematical Equation" for the Ensemble model you would like me to add as a final touch?** (I can render them in LaTeX or Mermaid blocks).
