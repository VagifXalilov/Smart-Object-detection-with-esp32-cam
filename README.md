# Smart-Object-detection-with-esp32-cam
# SmartDesk AI 🖥️👁️

**A hybrid AI + hardware system: an ESP32-CAM watches your desk, a YOLO model detects objects, and physical actuators react in real time.**

> Final project for the Data Science course / DIV Academy diploma.
> Author: Vagif — Automation Engineering student, ADNSU.

---

## 1. What does it do?

SmartDesk AI turns an ordinary desk into an AI-driven physical system.

| Detected situation | Physical reaction |
|---|---|
| Phone on the desk for more than N seconds | Buzzer beeps (focus alert) |
| Person present at the desk | Desk lamp relay ON |
| Nobody at the desk for a while | Desk lamp relay OFF (energy saving) |
| Cup detected, no change for a long time | Hydration reminder on dashboard |

Everything is logged to a CSV file so the data can be analysed with pandas (detections per hour, phone-usage time, etc.).

## 2. System architecture

```
┌────────────────┐   JPEG over WiFi (HTTP POST)   ┌──────────────────────┐
│  ESP32-CAM     │ ─────────────────────────────► │  Python server       │
│  (the "eyes")  │                                │  FastAPI + YOLOv8    │
│                │ ◄───────────────────────────── │  (the "brain")       │
│  buzzer, relay │   JSON: objects + actions      │  logging + dashboard │
│  (the "hands") │                                └──────────────────────┘
└────────────────┘
```

**Why this design?** The ESP32 has ~520 KB SRAM (+4 MB PSRAM), which is not enough for a full YOLO model at a usable frame rate. The camera node handles sensing and actuation; the heavy inference runs on a PC/laptop. A later stage compares this with fully on-device inference (Edge Impulse FOMO).

## 3. Hardware

| Part | Qty | Note |
|---|---|---|
| ESP32-CAM (AI-Thinker, OV2640) | 1 | Main board |
| ESP32-CAM-MB or FTDI (USB-to-serial) adapter | 1 | For flashing |
| 5 V / 2 A power supply | 1 | Weak USB power = random resets |
| Active buzzer | 1 | |
| 1-channel relay module (5 V) | 1 | Controls a desk lamp — **use low-voltage loads first** |
| Breadboard + jumper wires | 1 set | |
| (Optional) 2× SG90 servo | 2 | Pan-tilt tracking (stretch goal) |
| (Optional) SSD1306 OLED (I2C) | 1 | Local status display |

### Wiring

| ESP32-CAM pin | Connected to |
|---|---|
| GPIO 13 | Buzzer (+) |
| GPIO 14 | Relay IN |
| GPIO 4 | On-board flash LED (used as status LED) |
| 5V | Relay VCC, buzzer via GPIO |
| GND | Common ground |

> ⚠️ The ESP32-CAM exposes very few free GPIOs. Camera and SD card use most of them. GPIO 0 must be pulled to GND only while flashing, and GPIO 12 / GPIO 2 are strapping pins — avoid them for actuators.

## 4. Repository structure

```
smartdesk-ai/
├── firmware/
│   └── smartdesk_cam/
│       ├── smartdesk_cam.ino
│       └── config.h            # WiFi + server URL (do NOT commit real credentials)
├── server/
│   ├── app.py                  # FastAPI inference server
│   ├── rules.py                # Decision logic (debounce / hysteresis)
│   └── requirements.txt
├── training/
│   ├── data.yaml
│   └── train.py
├── notebooks/
│   ├── 01_dataset_eda.ipynb
│   ├── 02_model_evaluation.ipynb
│   └── 03_usage_analytics.ipynb
├── data/                       # (git-ignored) images + labels
├── models/                     # trained weights (best.pt)
├── docs/                       # wiring diagram, photos, demo GIF
├── .gitignore
└── README.md
```

