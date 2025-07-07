# 🤝 LeRobot Handshake Detection Project

<p align="center">
  <div style="display: flex; align-items: center; justify-content: center; gap: 20px;">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="media/lerobot-logo-thumbnail.png">
      <source media="(prefers-color-scheme: light)" srcset="media/lerobot-logo-thumbnail.png">
      <img alt="LeRobot Handshake Detection" src="media/lerobot-logo-thumbnail.png" style="height: 120px; width: auto;">
    </picture>
    <img alt="LeRobot Handshake Logo" src="media/lerobot-logo-handshake.png" style="height: 120px; width: auto;">
  </div>
  <br/>
  <br/>
</p>

<div align="center">

[![Python versions](https://img.shields.io/pypi/pyversions/lerobot)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/huggingface/lerobot/blob/main/LICENSE)
[![LeRobot](https://img.shields.io/badge/Built%20on-LeRobot-blue.svg)](https://github.com/huggingface/lerobot)

</div>

<h2 align="center">
    <p>🤖 Teaching SO-101 Robots to Shake Hands with Humans 🤝</p>
</h2>

<div align="center">
  <div style="display: flex; gap: 1rem; justify-content: center; align-items: center;" >
    <img
      src="media/so101/so101.webp?raw=true"
      alt="SO-101 performing handshake"
      title="SO-101 performing handshake"
      style="width: 40%;"
    />
    <img
      src="media/so101/so101-leader.webp?raw=true"
      alt="SO-101 leader for teleoperation"
      title="SO-101 leader for teleoperation"
      style="width: 40%;"
    />
  </div>

  <p><strong>Watch your SO-101 robot learn to detect handshake gestures and respond naturally!</strong></p>
  <p>Train it with your own demonstrations in minutes. 🚀</p>
  <p>Built on the powerful LeRobot framework with custom handshake detection. 🤝</p>
</div>

---

## 🎯 Project Overview

This project extends LeRobot to enable SO-101 robots to:

1. **🔍 Detect handshake gestures** using computer vision
2. **🎯 Recognize when humans are ready** to shake hands  
3. **🤖 Learn handshake motions** through teleoperation demonstrations
4. **🤝 Perform autonomous handshakes** with humans

### ✨ Key Features

- **Handshake Detection**: Real-time computer vision to detect when a person extends their hand
- **Single Camera Setup**: Optimized for front-facing camera (unlike standard SO-101 dual-camera setup)
- **Custom Recording Pipeline**: Specialized data collection for handshake interactions
- **Adaptive Training**: Policies that learn to correlate visual cues with appropriate actions
- **Safety-First**: Built-in safeguards and confidence thresholds

---

## 🛠️ Hardware Requirements

### Essential Hardware
- **2x SO-101 Robot Arms** (Leader + Follower) - [Build Guide](https://huggingface.co/docs/lerobot/so101)
  - Leader arm: For teleoperation during training
  - Follower arm: Performs the actual handshake
- **1x Camera** (USB webcam or similar)
  - Positioned to view the interaction area
  - 640x480 resolution minimum (higher resolution recommended)
- **Development Computer** with USB ports for robot arms and camera

### Software Requirements  
- **Python 3.10+**
- **LeRobot Framework** (included in this project)
- **OpenCV** for camera handling
- **MediaPipe** for hand detection (part of handshake detection module)

---

## 🚀 Quick Start

### 1. Installation

Clone this repository:
```bash
git clone https://github.com/Jackson1356/lerobot_handshake.git
cd lerobot_handshake
```

Create and activate a virtual environment:
```bash
conda create -y -n lerobot_handshake python=3.10
conda activate lerobot_handshake
```

Install dependencies:
```bash
conda install ffmpeg -c conda-forge
pip install -e .
```

### 2. Hardware Setup

1. **Build your SO-101 arms** following the [official guide](https://huggingface.co/docs/lerobot/so101)
2. **Connect your camera** and note its device index (usually 0)
3. **Find robot ports**:
   ```bash
   python lerobot/find_port.py
   ```
4. **Test camera connection**:
```bash
   python lerobot/find_cameras.py opencv
```

### 3. Calibrate Your Robots

Calibrate both leader and follower arms following the [official SO-101 calibration guide](https://huggingface.co/docs/lerobot/so101#calibrate):
```bash
# Calibrate follower arm
python lerobot/scripts/control_robot.py calibrate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0

# Calibrate leader arm  
python lerobot/scripts/control_robot.py calibrate \
    --robot.type=so101_leader \
    --robot.port=/dev/ttyACM1
```

### 4. Manual Teleoperation

Practice controlling your robot before recording datasets.

1. Without Camera
```bash
python -m lerobot.teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1
```

2. With Camera
```bash
python -m lerobot.teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --display_data=true
```

**Tips**: Start without camera first, practice smooth movements, always supervise the robot.

---


---

## 🎥 Recording Handshake Datasets

Use the specialized `record_handshake.py` script to collect demonstration data:

```bash
python lerobot/record_handshake.py \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --robot.id=follower_arm \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=leader_arm \
    --dataset.repo_id=your-username/handshake_dataset \
    --episode-time-s=30 \
    --reset-time-s=10 \
    --num-episodes=50
```

### Recording Process

1. **Handshake Detection Phase**: The system waits for a person to extend their hand
2. **Recording Phase**: You teleoperate the robot to perform the handshake
3. **Reset Phase**: Time to reset positions between episodes

### Keyboard Controls During Recording

- **⬅️ Left Arrow**: Re-record current episode (if something goes wrong)
- **➡️ Right Arrow**: End episode early and move to reset phase
- **⎋ Escape**: Stop entire recording session

---

## 📊 Visualizing Handshake Datasets

Before training your policy, it's recommended to visualize your recorded dataset to verify data quality and understand the handshake patterns. Use the HTML visualization tool to create an interactive web page:

```bash
python lerobot/scripts/visualize_dataset_html.py \
    --dataset.repo_id=your-username/handshake_dataset \
    --output-dir=./dataset_visualization
```

This will generate an HTML page with:
- **Episode thumbnails** showing key frames from each handshake
- **Interactive timeline** to scrub through episodes
- **Handshake detection visualization** overlaid on camera frames
- **Robot joint positions** synchronized with camera data
- **Episode statistics** including duration and detection rates

### Advanced Visualization Options

For more detailed analysis, you can also use the standard dataset visualization:

```bash
python lerobot/scripts/visualize_dataset.py \
    --dataset.repo_id=your-username/handshake_dataset \
    --episode-index=0 \
    --save-video
```

### Visualization Tips

1. **Check Detection Quality**: Verify that handshake detection is working correctly across episodes
2. **Review Robot Movements**: Ensure smooth teleoperation without jerky motions
3. **Validate Synchronization**: Confirm camera and robot data are properly aligned
4. **Identify Issues**: Look for episodes with poor lighting, occlusions, or failed detections

---

## 🎓 Training Handshake Policies

Train your robot using the specialized training script:

```bash
python lerobot/scripts/train_handshake.py \
    --dataset.repo_id=your-username/handshake_dataset \
    --policy.type=act \
    --policy.n_obs_steps=1 \
    --policy.chunk_size=100 \
    --training.lr=1e-5 \
    --training.batch_size=8 \
    --training.num_epochs=500 \
    --save.run_name=handshake_policy_v1
```

### Training Features

The training script includes handshake-specific metrics:
- **Detection Rate**: Percentage of frames where handshake was detected
- **Average Confidence**: Mean confidence score of handshake detection  
- **Standard Training Metrics**: Loss, gradient norms, learning rates

---

## 🎮 Running Trained Policies

Deploy your trained handshake policy:

```bash
python lerobot/scripts/eval.py \
    --policy.name=handshake_policy_v1 \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --eval.num_episodes=10
```

---

## 📁 Project Structure

```
lerobot_handshake/
├── lerobot/
│   ├── record_handshake.py         # 🎬 Custom recording script for handshake data
│   ├── scripts/
│   │   └── train_handshake.py      # 🎓 Custom training script with handshake metrics
│   └── common/
│       └── handshake_detection.py  # 🔍 Computer vision handshake detection
├── pose_detection/                 # 🧪 Development and testing scripts
│   ├── simple_camera_test.py       # Test camera functionality
│   └── test_handshake_detection_improved.py  # Test handshake detection
├── media/                          # 📸 Project media and documentation
└── README.md                       # 📖 This file
```

---

## ⚙️ Configuration

### Camera Configuration
The project uses a **single front-facing camera** (unlike standard SO-101 dual-camera setup):

```python
cameras = {
    "front": OpenCVCameraConfig(
        index_or_path=0,        # Usually 0 for built-in camera
        fps=30,                 # 30 FPS recommended
        width=640,              # Minimum resolution
        height=480,
        rotation=0              # Adjust if camera is rotated
    )
}
```

### Handshake Detection Parameters
Fine-tune detection sensitivity in `record_handshake.py`:

```python
# Confidence threshold for handshake detection (0.0-1.0)
handshake_confidence_threshold = 0.7

# Time to wait after detection before starting recording (seconds)  
handshake_detection_delay = 2.0

# Maximum time to wait for handshake detection (seconds)
handshake_timeout = 30.0
```

---

## 🔧 Troubleshooting

### Common Issues

**Camera not detected:**
```bash
# List available cameras
python lerobot/find_cameras.py opencv

# Test specific camera index
python pose_detection/simple_camera_test.py
```

**Robot connection issues:**
```bash
# Find robot ports
python lerobot/find_port.py

# Test robot connection
python lerobot/scripts/control_robot.py connect --robot.type=so101_follower --robot.port=YOUR_PORT
```

**Handshake detection not working:**
```bash
# Test detection algorithm
python pose_detection/test_handshake_detection_improved.py
```

### Performance Tips

1. **Lighting**: Ensure good lighting for reliable hand detection
2. **Camera Position**: Position camera to clearly see the handshake area
3. **Background**: Use plain backgrounds for better detection accuracy
4. **Distance**: Keep consistent distance between camera and interaction area

---

## 🤝 Contributing

This project builds on the amazing [LeRobot framework](https://github.com/huggingface/lerobot). 

### Development Workflow

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/improvement-name`
3. **Test your changes**: Run the test scripts in `pose_detection/`
4. **Submit pull request**: With clear description of changes

---

## 📚 Learn More

- **[LeRobot Documentation](https://huggingface.co/docs/lerobot)**: Core framework documentation
- **[SO-101 Build Guide](https://huggingface.co/docs/lerobot/so101)**: How to build your robot arms
- **[Camera Setup Guide](https://huggingface.co/docs/lerobot/cameras)**: Camera configuration and troubleshooting

---

## 🙏 Acknowledgments

- **[Hugging Face LeRobot Team](https://github.com/huggingface/lerobot)** for the incredible robotics framework
- **[TheRobotStudio](https://github.com/TheRobotStudio)** for the SO-101 robot design
- **MediaPipe Team** for robust hand detection algorithms
- **Computer Vision Community** for open-source pose detection research

---

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <p><strong>🤖 Happy Robot Building! 🤝</strong></p>
  <p>Train your SO-101 to shake hands with the world! 🌍</p>
</div>
