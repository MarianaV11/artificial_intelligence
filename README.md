# Project: AV2 IA — Banknote Authentication

## Description

This project implements a _Machine Learning_ model for **banknote authentication**.
The system reads a `.txt` dataset, splits it into training and testing sets, makes predictions, and generates evaluation metrics, including a **confusion matrix** saved as an image.

---

## Prerequisites

Before starting, install **uv**, a fast and modern Python package and environment manager.

### Installing uv

**Linux/macOS:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**

```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

Verify the installation:

```bash
uv --version
```

---

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/your-username/av2_ia.git
cd av2_ia
```

### 2. Create and activate the virtual environment

```bash
uv venv
source .venv/bin/activate   # Linux/macOS
# or
.venv\Scripts\activate      # Windows
```

### 3. Install and sync dependencies

Since this project uses a **`pyproject.toml`** file to manage dependencies, you can install everything with:

```bash
uv sync
```

This will automatically install all dependencies and ensure that your environment matches the project configuration.

If you add new dependencies later, use:

```bash
uv add NAME
```

---

## Running the Project

To execute the main script:

```bash
python main.py
```

---

## Project Structure

```
├── README.md
├── assets
│   ├── confusion_matrix_euclidean.png
│   └── confusion_matrix_manhatthan.png
├── data
│   └── data_banknote_authentication.txt
├── main.py
├── pyproject.toml
├── train
│   ├── __pycache__
│   │   ├── hold_out.cpython-310.pyc
│   │   ├── knn.cpython-310.pyc
│   │   └── metrics_graph.cpython-310.pyc
│   ├── hold_out.py
│   ├── knn.py
│   └── metrics_graph.py
└── uv.lock
```

---

## Expected Output

After running `main.py`, the program will generate:

- A **confusion matrix image** saved at `assets/matriz_confusao.png`
- Execution logs showing the data loading, training, and evaluation stages.

---

## Technologies Used

- Python 3.10+
- [uv](https://github.com/astral-sh/uv)
- NumPy
- Matplotlib
- Loguru

---
