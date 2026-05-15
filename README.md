# ODM-asi-camera-app
# AI SMART SURVEILLANCE SYSTEM

An advanced realtime AI-powered surveillance and inventory monitoring system built using YOLOv8, OpenCV, and Telegram integration.

------------------------------------------------------------

# PROJECT HIGHLIGHTS

## Realtime AI Object Detection
Implemented YOLOv8-based object detection capable of identifying multiple real-world objects in realtime through webcam/video input.

## Intelligent Object Monitoring
Built a dynamic monitoring engine that continuously tracks objects appearing and disappearing from the scene.

## Object Added Detection
Designed logic to detect when new objects enter the monitored environment and instantly trigger alerts.

## Object Removed Detection
Implemented realtime object removal monitoring to identify missing objects from the scene and notify the user immediately.

## Telegram Alert Integration
Integrated Telegram Bot API for instant remote notifications including:
- object added alerts
- object removed alerts
- evidence messages
- surveillance status updates

## Evidence Screenshot System
Created an automated screenshot capture system capable of storing:
- baseline scene image
- after-removal evidence image

Also added manual evidence sending controls.

## Interactive Surveillance UI
Developed a fullscreen futuristic surveillance interface using OpenCV with:
- live detection boxes
- realtime object counters
- event history panel
- warning banners
- control overlays

## Event Logging System
Implemented a realtime events panel that records:
- added objects
- removed objects
- evidence sending
- surveillance status changes

## Realtime Warning Banner
Created on-screen warning banners for immediate visual feedback whenever an object is added or removed.

## Lightweight Performance Optimization
Optimized the system for smoother realtime performance using:
- reduced processing resolution
- frame skipping
- threaded Telegram requests
- lightweight YOLO model

## Modular System Design
Structured the application into modular components including:
- detection engine
- event engine
- UI renderer
- Telegram service
- evidence manager

------------------------------------------------------------

# MAJOR TECHNICAL CHALLENGES FACED

## YOLO Detection Flickering
One of the biggest difficulties was handling unstable detections where objects would randomly disappear for a frame and reappear.

This caused:
- false alerts
- repeated notifications
- unstable tracking behavior

Solved using:
- confirmation timers
- debounce logic
- stable state management

------------------------------------------------------------

## Object Tracking Stability
Tracking realtime object additions/removals was difficult because YOLO detections constantly fluctuate due to:
- lighting changes
- object angle
- occlusion
- confidence variations

A custom event engine had to be developed to stabilize detection behavior.

------------------------------------------------------------

## Notification Spam Control
Early versions generated excessive Telegram alerts every frame.

Implemented:
- event confirmation timing
- alert state memory
- controlled notification triggering

to eliminate spam while keeping alerts responsive.

------------------------------------------------------------

## Multithreading Issues
Telegram requests initially caused UI lag and freezing because network calls blocked the main camera loop.

Solved by implementing:
- asynchronous threaded alerts
- threaded photo sending
- non-blocking event handling

------------------------------------------------------------

## Realtime UI Rendering
Rendering realtime detections, overlays, event logs, and warning banners smoothly while maintaining FPS required extensive optimization.

------------------------------------------------------------

## Detection State Management
Handling repeated object additions/removals correctly was difficult because:
- counts constantly changed
- objects flickered
- states got stuck

Required building:
- persistent object state logic
- confirmation timers
- reset mechanisms
- dynamic count tracking

------------------------------------------------------------

## Evidence Capture Synchronization
Capturing screenshots at the correct moment during object removal events required careful synchronization with the event engine.

------------------------------------------------------------

# TECHNOLOGIES USED

- Python
- YOLOv8
- OpenCV
- Ultralytics
- Telegram Bot API
- Multithreading

------------------------------------------------------------

# FINAL OUTCOME

The final system successfully performs:
- realtime AI surveillance
- object monitoring
- addition/removal detection
- remote alerting
- evidence generation
- interactive monitoring UI

with stable realtime performance and automated notification handling.

INSTALLATIONS

# TERMINAL INSTALLATION COMMANDS

# Create virtual environment
python -m venv venv

# Activate venv (Windows CMD)
venv\Scripts\activate

# Activate venv (PowerShell)
.\venv\Scripts\Activate.ps1

# Install required packages
pip install ultralytics
pip install opencv-python
pip install requests
pip install face-recognition
pip install numpy

# Install PyTorch
pip install torch torchvision torchaudio

# Optional EXE builder
pip install pyinstaller

# Run the project
python OD.py

# Optional EXE build
pyinstaller --onefile --windowed OD.py
