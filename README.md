# AI Drone — Emergency Detection & Search and Rescue

Hey! I'm building an AI-powered drone that can help find missing people and identify potential emergencies in cities, forests, mountains, and beaches.


The goal is to make search-and-rescue operations faster by combining **AI detection, autonomous pathfinding, and real-world emergency information**.

## What I've Built So Far

- Trained YOLO AI models to detect humans in aerial images
- Trained an emergency classifier to recognize fires, floods, collapsed buildings, and traffic accidents
- Used SARD 2, VisDrone, AFO, and AIDER datasets
- Tested models using precision, recall, and detection accuracy. (Some are low in accuracy, will need to retrain with more epochs)
- Used Kaggle GPUS for testing (I would reccomend, pretty fast and reliable)

## Building the Physical Drone

I'm also working on building a real quadcopter to eventually run my AI system. I currently have a self-built drone but need to implement the PI and the sensors.

My main components include:

| Component | Purpose |
|---|---|
| F450 Quadcopter Frame | Drone body |
| Pixhawk PX4 2.4.8 | Flight controller |
| Raspberry Pi 4 | AI processing |
| Camera Module | Capture images and videos |
| RS2205 2300KV Motors (×4) | Propulsion |
| 40A ESCs (×4) | Motor control |
| FlySky FS-i6 + FS-iA6B | Remote controller and receiver |
| LiPo Battery | Power |
| HC-SR04 Sensor | Distance sensing |

I'm working on integrating the Raspberry Pi, camera, and Pixhawk so the trained models can eventually work with the physical drone. I'm currently working on getting the drone to be stable during takeoff and actually fly. 

## My Research

**Research Question:** Can an uncertainty-guided search strategy locate simulated victims faster than a conventional fixed-grid drone search?

Planned three methods:

1. Fixed-Grid Search + A*
2. Bayesian Greedy Search + A*
3. Monte Carlo Tree Search (MCTS) + A*

I'll test which method finds simulated victims faster while considering obstacles, weather, and emergency-call information.

## How to Try It

Install Ultralytics:

```bash
pip install ultralytics
```

Run the pretrained person detector on an image named `sample.jpg`:

```bash
yolo predict model=yolo11n.pt source=sample.jpg classes=0 save=True
```

This saves an image with bounding boxes in `runs/detect/predict/`.

Custom training scripts are included in the repository.

## What's Next?

Workin toward combining all the AI models, using Seattle Fire 911 records and Open-Meteo weather data, testing autonomous search algorithms, and integrating everything with my physical drone.

Also, I'm planning on using a simulation to test the AI, alongside the final goal of actually integreting it on the physcial drone.

## Demo
<img src="./assets/0D373DD7-67C2-4BA6-B7E8-B789E245FCC6.jpeg">

<img src="./assets/Screenshot 2026-10-07 at 11.16.02 PM.png">

Demo URL if StarDance link is expired: https://streamable.com/lt0l03
---


