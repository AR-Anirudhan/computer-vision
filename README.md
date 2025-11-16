# Computer-Vision  Project
# Virtual Glasses Try-On 👓✨

Hello there! 👋 Welcome to the Virtual Glasses Try-On project.

This fun application uses your webcam to detect your face shape in real-time. Based on your shape (like oval, round, square, etc.), it recommends a style of glasses and lets you "try them on" virtually!

The glasses overlay will follow your face, turn with your head, and stay locked in place thanks to some smooth tracking magic.

*(Suggestion: Add a GIF or screenshot of your app working here!)*

---

## 🌟 Key Features

* **Real-time Face Shape Detection:** Automatically classifies your face into one of five shapes: Heart, Oblong, Oval, Round, or Square.
* **Smart Style Recommendations:** Get immediate feedback on which glasses suit you best (and which to avoid!) via a helpful sidebar.
* **Virtual Try-On Overlay:** See a PNG image of the glasses realistically overlaid on your face.
* **Silky-Smooth Tracking:** Uses **Kalman Filters** to smooth out jitter from the camera, making the glasses stick to your face naturally.
* **Interactive & Adjustable:**
    * Move the glasses up/down/left/right.
    * Make the glasses bigger or smaller.
    * Toggle the sidebar and face mesh on or off.

---

## 🚀 Getting Started

Here's how you can get the project running on your local machine.

### 1. Prerequisites

* Python 3.8+
* A webcam

### 2. Clone the Repository

First, get the code by cloning this repository:

```bash
git clone [https://github.com/your-username/your-project-name.git](https://github.com/your-username/your-project-name.git)
cd your-project-name
```

### 3. Set Up a Virtual Environment (Recommended)

It's always a good idea to keep your project's dependencies separate.

```bash
# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate
```

### 4. Install Dependencies

This project has two sets of requirements. You only need the first set to run the main try-on app.

**A) For the Main Try-On App (`app.py`):**

The core app only needs OpenCV, MediaPipe, and NumPy.

```bash
pip install opencv-python mediapipe numpy
```

**B) For All Developer Tools (like the labeling apps):**

If you also want to use the bonus developer tools (like `labelling_webapp.py`), install everything from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 5. ⚠️ Important: Add Glasses Images

The app needs transparent `.png` images of glasses to show you. You'll need to create an `assets` folder and add images in the correct subfolders.

Your folder structure **must** look like this:

```
your-project-folder/
│
├── assets/
│   ├── heart/
│   │   └── any_name_for_heart_glasses.png
│   ├── oblong/
│   │   └── any_name_for_oblong_glasses.png
│   ├── oval/
│   │   └── any_name_for_oval_glasses.png
│   ├── round/
│   │   └── any_name_for_round_glasses.png
│   └── square/
│       └── any_name_for_square_glasses.png
│
├── app.py
├── requirements.txt
└── ... (all other project files)
```

> **Note:** You can find transparent glasses PNGs on websites like [Pngtree](https://pngtree.com/), [FavPNG](https://favpng.com/), or [CleanPNG](https://www.cleanpng.com/).

### 6. Run the Application!

You're all set! Just run the `app.py` file:

```bash
python app.py
```

Look at your webcam, and the application will start!

---

## 🎮 How to Use the App (Controls)

Once the app is running, use these keys to control it:

| Key | Action |
| :--- | :--- |
| **`q`** | Quit the application. |
| **`s`** | Re-detect your face shape. |
| **`h`** | Toggle the info sidebar on/off. |
| **`m`** | Toggle the face mesh (the green lines) on/off. |
| **`1-5`** | Manually select a face shape (1=Heart, 2=Oblong, etc.). |
| | |
| **`w`** | Move glasses **UP**. |
| **`x`** | Move glasses **DOWN**. |
| **`a`** | Move glasses **LEFT**.POST. |
| **`d`** | Move glasses **RIGHT**. |
| **`c`** | Make glasses **BIGGER** (scale up). |
| **`z`** | Make glasses **SMALLER** (scale down). |
| **`r`** | **Reset** all adjustments to default. |

---

## 🧠 How It Works (A Peek Under the Hood)

1.  **Face Detection (MediaPipe):** The app uses `mediapipe.solutions.face_mesh` to find 478 unique 3D landmarks on your face in real-time.
2.  **Face Shape Classification (Geometric):** It doesn't use a heavy ML model for this. Instead, it calculates the **geometric ratios** between key landmarks (like jaw width vs. forehead width, and face height vs. face width). A set of `if/elif` rules then classifies the shape.
3.  **Position Smoothing (Kalman Filter):** Raw landmark data from a webcam can be "jittery." A `KalmanFilter` is used to predict where your eyes *should* be based on their previous positions. This creates a much smoother, more stable tracking effect.
4.  **Glasses Overlay (OpenCV):**
    * The app calculates the **distance** and **angle** between your eyes.
    * It resizes and rotates the glasses `.png` image to perfectly match that distance and angle.
    * Finally, it uses a function (`overlay_transparent`) to "paste" the glasses onto the video frame while respecting the image's transparency.

---

## 📁 Project File Guide

Here's a quick look at what each file does:

| File | Description |
| :--- | :--- |
| `app.py` | **This is the main application.** It contains all the code for the webcam feed, face detection, smoothing, and UI. This is the file you run! |
| `requirements.txt` | A list of all Python libraries used in the *entire* project (including bonus tools). |
| | |
| **Bonus Tools (For Developers)** | |
| `labelling_webapp.py` | A **Streamlit web app** for manually labeling a face dataset. A very modern and easy-to-use tool. Run with: `streamlit run labelling_webapp.py` |
| `labelling_app.py` | A **local (OpenCV)** version of the labeling app. It does the same thing as the Streamlit app but runs in a standard OpenCV window. Run with: `python labelling_app.py` |
| `clean_dataset.py` | A helper script to scan a dataset folder and remove any corrupted or invalid images before training a model. |
| | |
| **Component Files** | |
| `face_shape_detector.py` | A simplified, standalone version of the face shape detection logic. |
| `recommendation_engine.py` | A standalone class that just holds the recommendation text for each face shape. |
| `utils.py` | Contains helper functions like `overlay_transparent`. |

> **Note:** The main `app.py` is self-contained. It includes its *own* versions of the detection and recommendation logic, which are more advanced than the simple component files.

---

Hope you have fun with this project! Feel free to experiment with the adjustment values in `app.py` or add your own glasses.
