# SETUP

## Requirements

- Python 3.10 — this project requires Python 3.10.

## Installation

### 1. Clone/download the repository

Please make sure you have the full project folder with the following structure:

```
├── models/
├── notebooks/
├── src/
├── videos/
├── requirements.txt
```

### 2. Create a virtual environment

```bash
python3.10 -m venv venv
source venv/bin/activate # Mac/Linux
venv\Scripts\activate # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Dataset
Please download the zipped data here and unzip it in your repository: https://duke.box.com/s/cf8bjhzwov9t5q2133zgg7dgcvj8xw2r
Unfortunately due to Github file size constraints, the data could not be hosted in this repository.

## Running the Project

Once dependencies are installed, navigate to the `notebooks/` folder and open the relevant notebook.

## Notes

- All data is placed in the `data/` folder.
- Model architectures are saved to the `models/` folder.
- Source code and helper functions are in the`src/` folder.
