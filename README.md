# RockMassStr-Detector-Pro (RMSDP) v1.0

**RMSDP: A fast and open-source software for automated detection of rock joints**

This repository contains the **source code** for RMSDP.

This project is open-source and complies with academic standards for code availability.

---

## 📢 Download Executable (No Setup Required)

For users and reviewers who prefer a ready-to-run **version without** configuring a Python environment, we provide a standalone executable package (including all dependencies).

👉 **[Download the standalone executable (v1.0) from Releases](../../releases)**

> **Note:** The executable version is packaged as a `.zip` file in the **Releases** section to comply with repository file size limits.

---

## 🛠️ Environment Requirements

If you wish to run the source code directly, please ensure your environment meets the following specifications:

* **Python Version:** **3.8.10** (Strictly recommended for compatibility with TensorFlow and Open3D).
* **OS:** Windows 10/11 (Recommended).

---

## 🚀 Installation & Usage

Follow these steps to set up the project source code on your local machine.

### Clone the Repository and Install Dependencies

Clone the source code to your local machine and install all required Python libraries using the provided requirements file.

```bash
# 1. Clone the repository
git clone [https://github.com/clouder4lucker-gif/RockMassStr-Detector-Pro-RMSDP.git](https://github.com/clouder4lucker-gif/RockMassStr-Detector-Pro-RMSDP.git)

# 2. Enter the project directory
cd RockMassStr-Detector-Pro-RMSDP

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the main program
python Operation.py

```

## 📂 Repository Structure
* **Operation.py:** The main entry point of the application.
* **ui_Main_Window.py:** Implementation of the point cloud processing interface.
* **ui_Login_2.py:** Implementation of the user login interface.
* **Catagory_rc.py:** The implementation files of the icons in the point cloud processing interface.
* **Insert_||—_rc.py:** The implementation files of the icons in the user login interface.
* **requirements.txt:** List of Python dependencies (generated via pip freeze).
