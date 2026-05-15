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

# HOW TO USE

1. Run the project:

python OD.py

------------------------------------------------------------

2. Wait for baseline initialization.

The system will automatically:
- start the camera
- load YOLO model
- initialize surveillance
- save the baseline scene

You will receive:

✅ SURVEILLANCE ACTIVE

on Telegram once monitoring begins.

------------------------------------------------------------

3. Object Monitoring

The system continuously monitors:
- objects entering the scene
- objects removed from the scene
- people appearing/disappearing
- known and unknown faces

------------------------------------------------------------

4. Face Recognition

To register known faces:

Place face images inside:

faces/known/

Example:

faces/known/Alice.jpg
faces/known/Bob.jpg

The system automatically:
- loads known faces
- assigns stable IDs
- recognizes returning persons

Unknown people are automatically stored as:

Intruder-P1
Intruder-P2
etc.

------------------------------------------------------------

5. Telegram Alerts

The system sends realtime Telegram notifications for:
- new person detected
- person returned
- person left
- object added
- object removed
- evidence screenshots
- surveillance status updates

------------------------------------------------------------

6. Evidence Screenshots

The system stores:
- baseline image
- after-event image

Manual evidence can be sent using keyboard controls.

------------------------------------------------------------

7. Controls

L → Toggle left information panel

R → Toggle events panel

D → Toggle detection boxes

B → Toggle footer

S → Send evidence screenshots to Telegram

Q → Quit application

------------------------------------------------------------

8. Logs

All events are automatically stored inside:

logs/

Generated logs:
- surveillance_log.txt
- events_table.txt
- faces_db.json

------------------------------------------------------------

9. Captured Faces

Detected faces are stored inside:

faces/captured/

Known faces are stored inside:

faces/known/

------------------------------------------------------------

10. Performance Notes

For better performance:
- use NVIDIA GPU if available
- close unnecessary background apps
- ensure good lighting conditions
- use stable camera positioning

------------------------------------------------------------

11. Detection Notes

The system performs best when:
- objects are clearly visible
- camera remains stable
- lighting is consistent
- objects are not heavily occluded

------------------------------------------------------------

12. Stopping the System

Press:

Q

to safely close the surveillance system.

# AI SMART SURVEILLANCE SYSTEM

An advanced realtime AI-powered surveillance and monitoring system built using YOLOv8, OpenCV, Face Recognition, and Telegram integration.

------------------------------------------------------------

# WHAT THIS SYSTEM DOES

## Realtime Object Detection
Detects multiple realtime objects through webcam/video feed using YOLOv8.

Examples:
- person
- bottle
- laptop
- chair
- phone
- backpack
- monitor
- keyboard
- and many more

------------------------------------------------------------

## Realtime Object Monitoring
Continuously monitors the environment and detects:
- objects entering the scene
- objects removed from the scene
- sudden environment changes

------------------------------------------------------------

## Object Added Alerts
Whenever a new object appears:
- warning banner appears on screen
- Telegram notification is sent
- event gets logged

------------------------------------------------------------

## Object Removed Alerts
Whenever an object disappears:
- warning banner appears
- removal alert is sent
- evidence image is captured
- event is added to history panel

------------------------------------------------------------

## Face Recognition System
Supports known and unknown face detection.

Known faces:
- automatically recognized
- assigned names
- tracked when returning

Unknown faces:
- automatically registered
- assigned unique IDs
- stored for future monitoring

------------------------------------------------------------

## Intruder Detection
Unknown people entering the monitored area are automatically identified and logged.

------------------------------------------------------------

## Telegram Integration
Realtime Telegram alerts for:
- object added
- object removed
- intruder detected
- person returned
- evidence screenshots
- surveillance startup status

------------------------------------------------------------

## Evidence Screenshot System
Captures:
- baseline scene
- event screenshots
- after-removal evidence

Can also manually send screenshots directly to Telegram.

------------------------------------------------------------

## Event History Panel
Displays live events such as:
- object added
- object removed
- face detected
- evidence sent
- surveillance events

------------------------------------------------------------

## Warning Banner System
Displays large realtime warning messages directly on screen whenever important events occur.

------------------------------------------------------------

## Fullscreen Surveillance UI
Custom futuristic surveillance interface with:
- realtime detection boxes
- object counters
- live event logs
- overlays
- fullscreen monitoring mode

------------------------------------------------------------

## Optimized Realtime Performance
Performance optimized using:
- frame skipping
- lightweight YOLO model
- threaded Telegram requests
- reduced processing resolution

------------------------------------------------------------

# HOW TO USE

## 1. Run The Project

Run:

python OD.py

------------------------------------------------------------

## 2. System Startup

The system automatically:
- opens webcam
- loads YOLO model
- initializes face recognition
- starts surveillance engine
- saves baseline scene

Telegram receives:

✅ SURVEILLANCE ACTIVE

------------------------------------------------------------

## 3. Object Monitoring

The system continuously checks:
- what objects are visible
- what disappeared
- what newly appeared

------------------------------------------------------------

## 4. Face Registration

Place known face images inside:

faces/known/

Example:

faces/known/Alice.jpg
faces/known/Bob.jpg

The system automatically loads them during startup.

------------------------------------------------------------

## 5. Unknown Face Handling

Unknown people are automatically:
- detected
- assigned IDs
- stored inside:

faces/captured/

------------------------------------------------------------

## 6. Telegram Notifications

You receive alerts for:
- object added
- object removed
- unknown person detected
- known person returned
- evidence screenshots

------------------------------------------------------------

## 7. Evidence Screenshots

Press:

S

to manually send:
- baseline image
- latest evidence image

to Telegram.

------------------------------------------------------------

## 8. Controls

L → Toggle left information panel

R → Toggle events panel

D → Toggle detection boxes

B → Toggle footer

S → Send evidence screenshots

Q → Quit application

------------------------------------------------------------

## 9. Logs

All events are stored automatically inside:

logs/

Examples:
- surveillance_log.txt
- event_history.txt
- face_database.json

------------------------------------------------------------

## 10. Performance Tips

For best performance:
- use NVIDIA GPU if available
- maintain stable lighting
- avoid shaky camera movement
- close unnecessary applications

------------------------------------------------------------

## 11. Recommended Hardware

Recommended:
- 8GB+ RAM
- NVIDIA GPU
- modern CPU
- HD webcam

------------------------------------------------------------

## 12. Closing The System

Press:

Q

to safely terminate the surveillance system.
