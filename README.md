This project demonstrates the application of automated layout analysis and deep learning models to process and extract information from historical newspaper documents. The case study focuses on the Austrian newspaper *Neue Kronen Zeitung*, dated August 22nd, 1989.


AUTOMATED LAYOUT ANALYSIS AND DEEP LEARNING MODELS FOR PROCESSING OF HISTORICAL NEWSPAPER DOCUMENTS.
A CASE STUDY OF "NEUE KRONEN ZEITUNG" AUGUST 22ND, 1989

## Overview

Historical newspapers hold valuable information for historians and digital humanities researchers but often feature complex page layouts combining headlines, paragraphs, images, and illustrations. Automated layout analysis can help efficiently classify and process these different sections for further study.

This project explores:
- Testing multiple pre-trained deep learning models from the Layoutparser Model Zoo (using Detectron2) for layout detection.
- Applying Optical Character Recognition (OCR) via Tesseract for extracting text from identified layout regions.
- Automated batch-processing of multiple newspaper pages for efficient text and image extraction.

---

## Dataset

- **Source:** Scanned images of 12 pages from the *Neue Kronen Zeitung* newspaper, dated August 22, 1989.
- **Image resolution:** Ranging from 2080 to 2900 pixels.
- **Capture method:** Samsung Galaxy S20 camera scanning with PDF Scan app.

---

## Tools and Libraries

- **Detectron2:** State-of-the-art computer vision library developed by Facebook AI for object detection and segmentation.
- **Layoutparser:** Python library for document layout analysis that integrates Detectron2 pre-trained models.
- **Tesseract OCR:** Open-source OCR engine for recognizing text in images.
- **Python 3.8 environment** (recommended via Anaconda)
- Ubuntu LTS on Windows (WSL) recommended for smoother setup.

---

## Tested Models

1. **PublayNet**
- Designed for document layout of scientific papers.
- Struggled with newspaper layouts, causing noisy and inaccurate results at default thresholds.

2. **NewspaperNavigator**
- Trained on historic newspapers from WW1 era focused on detecting photographs, illustrations, headlines, and ads rather than text paragraphs.
- Less accurate for modern and more complex newspaper layouts.

3. **PrimaLayout** *(Best Performance)*
- Contains diverse documents including newspapers and magazines.
- Strong generalization and flexible enough for newspaper layouts.
- Low noise, good for extracting text and image regions. Minor overlapping in headlines observed.

---

## Key Features

- **Automated layout detection:** Segment pages into TextRegions, ImageRegions, and other logical structures.
- **Focused extraction:** Filter and extract specific regions like images or text blocks.
- **OCR integration:** Use Tesseract with German language data to extract readable text from cropped text regions.
- **Batch processing:** Automate processing of multiple newspaper pages, saving extracted text and images in an organized file structure.

---

## Setup Instructions

1. Install Ubuntu LTS environment (via WSL on Windows or native Linux recommended).
2. Install Anaconda and create a Python 3.8 environment.
3. Install Layoutparser and Detectron2 along with necessary dependencies via pip:

```bash
pip install layoutparser torchvision
pip install detectron2@git+https://github.com/facebookresearch/detectron2.git@v0.5#egg=detectron2

1. Download Tesseract OCR and German tessdata files from Tesseract GitHub repository.

2. Set environment variable for Tesseract data path:

os.environ['TESSDATA_PREFIX'] = '/usr/share/tesseract-ocr/4.00/tessdata/'

Usage Example

import layoutparser as lp
from PIL import Image
import os

# Load PrimaLayout model (best performance)
model = lp.Detectron2LayoutModel(
config_path='lp://PrimaLayout/mask_rcnn_R_50_FPN_3x/config',
label_map={1: "TextRegion", 2: "ImageRegion", 3: "TableRegion", 4: "MathsRegion", 5: "SeparatorRegion", 6: "OtherRegion"},
extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5]
)

# Load an example newspaper page
image_path = os.path.join("images", "kz-007.png")
image = Image.open(image_path)

# Detect layout
layout_result = model.detect(image)

# Filter text regions for OCR
text_blocks = lp.Layout([b for b in layout_result if b.type == 'TextRegion'])

# Initialize Tesseract OCR agent
ocr_agent = lp.TesseractAgent(languages='deu')

# Extract text from each region
for block in text_blocks:
segment_image = block.pad(left=15, right=15, top=5, bottom=5).crop_image(image)
text = ocr_agent.detect(segment_image)
block.set(text=text, inplace=True)
print(block.text)

Results and Findings

- PrimaLayout provided the most accurate segmentation for historical newspaper pages.

- Smaller, simpler paragraphs were accurately OCR'd by Tesseract; complex headers and mixed fonts posed challenges.

- Batch processing allowed automated extraction of multiple pages efficiently.

- Demonstrates potential for aiding historians and digital humanists to analyze complex newspaper layouts at scale.

References

- Antonacopoulos et al., "A Realistic Dataset for Performance Evaluation of Document Layout Analysis"

- Lee et al., "The Newspaper Navigator Dataset: Extracting And Analyzing Visual Content from 16 Million Historic Newspaper Pages"

- Winastwan, R., "Analyzing Document Layout with LayoutParser"

- Wu et al., "Detectron2"

(See full references in the project report.)

License

(Specify your license here, e.g., MIT License)

Contact

For questions and contributions, please contact Leonardo Villarroel ([contact info if desired]).

This project was part of the MA Digital Humanities course at the University of Vienna under Prof. Xung Gong (September 2023).
