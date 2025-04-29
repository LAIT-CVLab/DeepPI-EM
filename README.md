# DeepPI-EM: Deep learning-driven automated mitochondrial segmentation for analysis of complex transmission electron microscopy images
Authors : Chan Jang, Hojun Lee, [Jaejun Yoo](https://scholar.google.co.kr/citations?hl=en&user=7NBlQw4AAAAJ), [Haejin Yoon](https://scholar.google.co.kr/citations?user=1paFUdEAAAAJ&hl=en&oi=ao)

## Abstract
> Mitochondria are central to cellular energy production and regulation, with their morphology tightly linked to functional performance. Precise analysis of mitochondrial ultrastructure is crucial for understanding cellular bioenergetics and pathology. While transmission electron microscopy (TEM) remains the gold standard for such analyses, traditional manual segmentation methods are time-consuming and prone to error. In this study, we introduce a novel deep learning framework that combines probabilistic interactive segmentation with automated quantification of mitochondrial morphology. Leveraging uncertainty analysis and real-time user feedback, the model achieves comparable segmentation accuracy while reducing analysis time by 90% compared to manual methods. Evaluated on both benchmark Lucchi++ datasets and real-world TEM images of mouse skeletal muscle, the pipeline not only improved efficiency but also identified key pathological differences in mitochondrial morphology between wild-type and mdx mouse models of Duchenne muscular dystrophy. This automated approach offers a powerful, scalable tool for mitochondrial analysis, enabling high-throughput and reproducible insights into cellular function and disease mechanisms.

## Repository Structure

- `pi_seg/`: Contains core modules for image segmentation and processing.
- `config.py`: Configuration settings for model training and evaluation.
- `dataset.py`: Scripts for dataset loading and preprocessing.
- `main.py`: Entry point for training and evaluating the model.
- `model.py`: Definitions of the neural network architectures used.
- `test.py`: Scripts for testing the trained models.
- `train_validate.py`: Procedures for training and validating the models.
- `gui/`: Contains interactive segmentation logic and user interface components.
- `run_gui.py`: Launches the graphical user interface for interactive image annotation and analysis.

## Usage
### Installation

To set up the environment, follow these steps:

#### 1️⃣ Clone the repository

```bash
git clone https://github.com/LAIT-CVLab/DeepPI-EM.git
cd DeepPI-EM
```

#### 2️⃣ Create a virtual environment (optional but recommended)
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

#### 3️⃣ Install the required packages
If you are using **CPU** or already have a compatible CUDA setup, install the default requirements:

```bash
pip install -r requirements.txt
```

If you are using CUDA 11.3 with PyTorch 1.11.0, install the CUDA-compatible version instead:
```bash
pip install -r requirements-cu113.txt
```
> `requirements-cu113.txt` includes dependencies optimized for CUDA 11.3, such as `torch==1.11.0+cu113` and `mmcv-full`, which are not available on the default PyPI index.


### Training & Evaluation
All training and testing parameters can be configured in [`config.py`](config.py). And pre-trained models required for training and evaluation can be downloaded from the following links:
  - [CEM500k pre-trained model](https://github.com/volume-em/cem-dataset)
  - [Our model (DeepPI-EM)](https://drive.google.com/drive/folders/1n2ZqbJEHPyMB_6a6OTBBACt5Jct2PZJc?usp=sharing)

#### Training
```bash
python main.py
```

#### Evaluating
```bash
python test.py
```


### Graphical User Interface (GUI)
We provide a simple GUI for evaluating the trained model and enabling practical interaction. A demo using the Lucchi++ dataset is available **[here](https://b17f47699f8116e70e.gradio.live)**.  
Users can specify any trained model within [`run_gui.py`](run_gui.py) and launch the interface as follows:
```bash
python run_gui.py
```

## Dataset

This project uses publicly available **electron microscopy datasets**:
- [Lucchi++ Dataset](https://casser.io/connectomics/)
- [Kasthuri Dataset](https://casser.io/connectomics/)
- [MitoEM-H](https://mitoem.grand-challenge.org/)
- [Skeletal Muscle TEM Dataset](https://drive.google.com/drive/folders/1n2ZqbJEHPyMB_6a6OTBBACt5Jct2PZJc?usp=sharing)
  : A custom dataset developed by our team for mitochondria segmentation in skeletal muscle transmission electron microscopy (TEM) images.  

📌 **Note**: These datasets are publicly accessible and can be used for research purposes.  
