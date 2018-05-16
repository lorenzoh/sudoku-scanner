# sudoku-scanner
[![Build Status](https://travis-ci.org/lorenzoh/sudoku-scanner.svg?branch=master)](https://travis-ci.org/lorenzoh/sudoku-scanner)

A scanner that detects, recognizes and solves sudoku puzzles from real-life pictures using a pipeline of classical Computer Vision algorithms.


# How to use

Install dependencies (use Python 3.6):

```pip install -r requirements.txt```

Run on image:

```python src/sudoku-scanner.py <path to your image>```

For some more options, use:

```python src/sudoku-scanner.py --help```

## Pipeline
The pipeline is divided into 5 main steps

You can generate the following images for your own image by following the steps in ```/notebooks/VisualizePipeline.ipynb```

### Input

![Input image](https://github.com/lorenzoh/sudoku-scanner/raw/master/tests/testing_data/image1000.jpg)

### 1. Preprocessing input image

![Image after preprocessing](https://github.com/lorenzoh/sudoku-scanner/raw/master/images/step0.jpg)

### 2. Exposing the grid

![Exposed grid](https://github.com/lorenzoh/sudoku-scanner/raw/master/images/step1.jpg)

### 3. Finding corners and transforming perspective

![Transformed grid](https://github.com/lorenzoh/sudoku-scanner/raw/master/images/step2.jpg)

### 4. Preprocessing grid 

![Preprocessed grid](https://github.com/lorenzoh/sudoku-scanner/raw/master/images/step3.jpg)

### 5. Digit cleaning and classification

The individual digits are then cut from the grid, cleaned individually and classified by a small ConvNet.







