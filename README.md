🚗 Driver Drowsiness Detection

📌 Overview

Driver drowsiness is a major cause of road accidents. This project uses computer vision and deep learning to detect driver fatigue in real-time and alert them before an accident occurs.

🛠 Features

Real-Time Face & Eye Detection using OpenCV & Haar Cascades

Deep Learning Model for accurate drowsiness detection

Audio Alert System to wake up the driver

Efficient & Lightweight for real-world applications

🔧 Installation

1️⃣ Clone the Repository

git clone https://github.com/YuvanshDua/Driver-Drowsiness.git
cd Driver-Drowsiness

2️⃣ Install Dependencies

Make sure you have Python installed, then run:

pip install -r requirements.txt

3️⃣ Run the Application

python Driver\ Face\ detection.py

📂 Project Structure

Driver-Drowsiness/
│-- data/                # Training & validation datasets
│-- detected/            # Folder to store detected faces
│-- haar cascade files/  # Haar cascade XML files for face & eye detection
│-- models/              # Pre-trained deep learning models
│-- myenv/               # Virtual environment (excluded from Git)
│-- abc.py               # Supporting script
│-- alarm.wav            # Alarm sound file
│-- Driver Face detection.py  # Main script
│-- image.jpg            # Sample image
│-- model.py             # Model training script
│-- README.md            # Project documentation
│-- requirements.txt     # Python dependencies

🖥 Technologies Used

Python

OpenCV for face & eye detection

TensorFlow/Keras for deep learning

NumPy & Pandas for data handling

Pygame for playing alarm sound

🎯 Future Enhancements

Improve model accuracy with more diverse datasets

Implement real-time video streaming for better performance

Add mobile app integration for wider accessibility

🤝 Contributing

Feel free to contribute to this project! 🚀

Fork the repo

Create a new branch

Commit your changes

Submit a pull request

💡 **Made with ❤️ by **YuvanshDua

