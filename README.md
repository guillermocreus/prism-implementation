# PRISM implementation

Supervised and Experiential Learning Lab 1 - MSc in AI at UPC

## Introduction

## Running the code

## Authors

Guillermo Creus


# PRISM implementation

Practical work from the Supervised and Experiential Learning course - MSc in AI at UPC

## Introduction

The purpose of this report is to gain insights on rule basedclassifiers. In particular, this report will be centered on the implementation of PRISM. Based on the work of Jadzia Cendrowska, the PRISM algorithm will be developed, improved, and evaluated on three datasets from the UC Irvine Machine Learning Repository.

The motivation of this work is double sided. On one hand, academically, it is highly stimulating to implement from scratch an algorithm and have full control over it, where the opportunities to learn are endless. In addition, the mistakes on the way will be key at mastering this algorithm, contrary to other ready-to-use solutions.

On the other hand, by developing this algorithm one has the opportunity to test its performance on datasets and analyze where it fails, so as to improve it in the future.

As a side note, PRISM will be evaluated on three datasets of different size in order to determine how it scales. It is relevant to check if rule-based classifiers' performance is independent of the size and dataset used.

## Report

A more detailed analysis of this task is shown in this [document](./report.pdf).

## Instructions

Steps to run the script

### Running script for the first time
This section shows how to create a virtual environment, which is needed to run the scripts from this project
1. Open folder in terminal
```bash
cd <root_folder_of_project>/
```
2. Create virtual env
```bash
python3 -m venv venv/
```
3. Open virtual env
```bash
source venv/bin/activate
```
4. Install required dependencies
```bash
pip install -r requirements.txt
```
One can check if dependencies were installed by running the following command
```bash
pip list
```

1. Close virtual env
```bash
deactivate
```

### Execute scripts

1. Open virtual env
```bash
source venv/bin/activate
```
2. Running the script

	2.1.  For **Wines dataset** (small dataset) execute
	```bash
	python3 main.py small
	```

	2.2. For **Breast cancer Wisconsin** (medium dataset) execute
	```bash
	python3 main.py small
	```
 	2.3 For **Seismic bumps** (large dataset) execute
	```bash
	python3 main.py large
	```

3. Close virtual env
```bash
deactivate
```

## Authors

Guillermo Creus