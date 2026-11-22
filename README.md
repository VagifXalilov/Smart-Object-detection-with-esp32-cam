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

## 5. Step-by-step tutorial

### Step 1 — Test the camera

1. Install **Arduino IDE 2.x**.
2. `File → Preferences → Additional boards manager URLs`, add:
   `https://espressif.github.io/arduino-esp32/package_esp32_index.json`
3. `Tools → Board → Boards Manager` → install **esp32 by Espressif**.
4. Select **AI Thinker ESP32-CAM**, enable **PSRAM**.
5. Open `File → Examples → ESP32 → Camera → CameraWebServer`, enter your WiFi, upload, and open the IP in a browser.

✅ Checkpoint: you see a live video stream.

### Step 2 — Set up the Python server

```bash
git clone https://github.com/<your-username>/smartdesk-ai.git
cd smartdesk-ai/server
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

`requirements.txt`:

```
fastapi
uvicorn
ultralytics
opencv-python
numpy
pandas
```

Find your PC's local IP (`ipconfig` / `ifconfig`) — the ESP32 will send frames there. PC and ESP32 must be on the same WiFi network.

### Step 3 — Inference server (`server/app.py`)

```python
import csv, time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import Response
from ultralytics import YOLO

from rules import DeskState

app = FastAPI()
model = YOLO("../models/best.pt" if Path("../models/best.pt").exists() else "yolov8n.pt")
state = DeskState()
latest_jpg = b""

LOG = Path("../data/detections_log.csv")
if not LOG.exists():
    LOG.parent.mkdir(exist_ok=True)
    LOG.write_text("timestamp,label,confidence,latency_ms\n")


@app.post("/detect")
async def detect(request: Request):
    global latest_jpg
    data = await request.body()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "bad image"}

    t0 = time.time()
    result = model.predict(img, conf=0.4, verbose=False)[0]
    latency_ms = round((time.time() - t0) * 1000, 1)

    labels = [model.names[int(c)] for c in result.boxes.cls]
    confs = [float(c) for c in result.boxes.conf]

    with LOG.open("a", newline="") as f:
        w = csv.writer(f)
        for lab, cf in zip(labels, confs):
            w.writerow([time.time(), lab, round(cf, 3), latency_ms])

    _, buf = cv2.imencode(".jpg", result.plot())
    latest_jpg = buf.tobytes()

    actions = state.update(set(labels))
    return {"objects": labels, "latency_ms": latency_ms, **actions}


@app.get("/latest.jpg")
def latest():
    return Response(content=latest_jpg, media_type="image/jpeg")
```

### Step 4 — Decision logic (`server/rules.py`)

Raw detections flicker from frame to frame. Acting on a single frame causes false alarms, so the logic uses **debouncing** (N consecutive frames) and **hysteresis** (different thresholds to turn ON vs OFF) — the same idea used in control systems.

```python
class DeskState:
    def __init__(self, on_frames=3, off_frames=20, phone_frames=10):
        self.on_frames = on_frames        # frames to confirm presence
        self.off_frames = off_frames      # frames of absence before lamp OFF
        self.phone_frames = phone_frames  # frames of phone before alert
        self.person_count = 0
        self.absent_count = 0
        self.phone_count = 0
        self.lamp = False

    def update(self, labels: set) -> dict:
        # person presence with hysteresis
        if "person" in labels:
            self.person_count += 1
            self.absent_count = 0
        else:
            self.absent_count += 1
            self.person_count = 0

        if self.person_count >= self.on_frames:
            self.lamp = True
        elif self.absent_count >= self.off_frames:
            self.lamp = False

        # phone alert with debouncing
        self.phone_count = self.phone_count + 1 if "cell phone" in labels else 0
        return {"lamp": self.lamp, "phone_alert": self.phone_count >= self.phone_frames}
```

### Step 5 — Firmware (`firmware/smartdesk_cam/smartdesk_cam.ino`)

Libraries: **ArduinoJson** (install from Library Manager).

`config.h`:

```cpp
#pragma once
#define WIFI_SSID  "your-wifi"
#define WIFI_PASS  "your-password"
#define SERVER_URL "http://192.168.1.50:8000/detect"   // your PC IP
```

`smartdesk_cam.ino`:

```cpp
#include "esp_camera.h"
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include "config.h"

// AI-Thinker ESP32-CAM pin map
#define PWDN_GPIO_NUM 32
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 0
#define SIOD_GPIO_NUM 26
#define SIOC_GPIO_NUM 27
#define Y9_GPIO_NUM 35
#define Y8_GPIO_NUM 34
#define Y7_GPIO_NUM 39
#define Y6_GPIO_NUM 36
#define Y5_GPIO_NUM 21
#define Y4_GPIO_NUM 19
#define Y3_GPIO_NUM 18
#define Y2_GPIO_NUM 5
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM 23
#define PCLK_GPIO_NUM 22

#define BUZZER_PIN 13
#define RELAY_PIN  14

void initCamera() {
  camera_config_t c;
  c.ledc_channel = LEDC_CHANNEL_0;
  c.ledc_timer = LEDC_TIMER_0;
  c.pin_d0 = Y2_GPIO_NUM; c.pin_d1 = Y3_GPIO_NUM; c.pin_d2 = Y4_GPIO_NUM;
  c.pin_d3 = Y5_GPIO_NUM; c.pin_d4 = Y6_GPIO_NUM; c.pin_d5 = Y7_GPIO_NUM;
  c.pin_d6 = Y8_GPIO_NUM; c.pin_d7 = Y9_GPIO_NUM;
  c.pin_xclk = XCLK_GPIO_NUM; c.pin_pclk = PCLK_GPIO_NUM;
  c.pin_vsync = VSYNC_GPIO_NUM; c.pin_href = HREF_GPIO_NUM;
  c.pin_sccb_sda = SIOD_GPIO_NUM; c.pin_sccb_scl = SIOC_GPIO_NUM;
  c.pin_pwdn = PWDN_GPIO_NUM; c.pin_reset = RESET_GPIO_NUM;
  c.xclk_freq_hz = 20000000;
  c.pixel_format = PIXFORMAT_JPEG;
  c.frame_size = FRAMESIZE_VGA;     // 640x480
  c.jpeg_quality = 12;              // lower = better quality, bigger file
  c.fb_count = 2;
  c.grab_mode = CAMERA_GRAB_LATEST;
  if (esp_camera_init(&c) != ESP_OK) {
    Serial.println("Camera init failed");
    while (true) delay(1000);
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(RELAY_PIN, OUTPUT);
  initCamera();

  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) { delay(500); Serial.print("."); }
  Serial.println("\nWiFi connected: " + WiFi.localIP().toString());
}

void loop() {
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) { delay(100); return; }

  HTTPClient http;
  http.begin(SERVER_URL);
  http.addHeader("Content-Type", "image/jpeg");
  int code = http.POST(fb->buf, fb->len);

  if (code == 200) {
    JsonDocument doc;
    if (!deserializeJson(doc, http.getString())) {
      digitalWrite(RELAY_PIN, doc["lamp"] ? HIGH : LOW);   // flip if your relay is active-low
      digitalWrite(BUZZER_PIN, doc["phone_alert"] ? HIGH : LOW);
    }
  } else {
    Serial.printf("HTTP error: %d\n", code);
  }
  http.end();
  esp_camera_fb_return(fb);
  delay(200);   // ~2-4 FPS is enough for a desk
}
```

✅ Checkpoint: hold a phone in front of the camera → buzzer sounds after ~10 frames.

### Step 6 — Build your own dataset (the Data Science part)

Pretrained YOLO is a good start, but a **custom dataset from the ESP32-CAM itself** is what makes this a real data science project (the OV2640 sensor has different noise, colour and distortion than the images YOLO was pretrained on).

1. Collect **150–300 images per class** from your real desk — vary light (day/night/lamp), angle, clutter, distance.
2. Label with **Roboflow** or **LabelImg** (YOLO format).
3. Split **70 / 20 / 10** (train / val / test). Split by *session*, not randomly, to avoid near-duplicate leakage.
4. Apply augmentation (brightness, blur, flip, mosaic).

`training/data.yaml`:

```yaml
path: ../data/dataset
train: images/train
val: images/val
test: images/test
names:
  0: phone
  1: cup
  2: laptop
  3: book
  4: person
```

### Step 7 — Train and evaluate

`training/train.py`:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")            # transfer learning
model.train(data="data.yaml", epochs=50, imgsz=640, batch=16, patience=10)
metrics = model.val(split="test")
print(metrics.box.map50, metrics.box.map)
```

Report these metrics in `notebooks/02_model_evaluation.ipynb`:

| Metric | Formula |
|---|---|
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| F1 | 2·P·R / (P + R) |
| mAP@0.5 | mean of per-class Average Precision at IoU 0.5 |
| Latency | end-to-end ms per frame (camera → server → actuator) |

Also include: confusion matrix, per-class error analysis, and examples of failure cases.

### Step 8 — Experiments (what makes it a strong project)

| Experiment | Question |
|---|---|
| Pretrained vs. custom-trained YOLOv8n | How much does domain-specific data help? |
| 320 vs. 640 input size | Accuracy vs. latency trade-off |
| Good light vs. dim light | How robust is the model? |
| Server inference vs. Edge Impulse FOMO on ESP32 | Cloud/PC vs. edge trade-off |

### Step 9 — Analytics notebook

Load `data/detections_log.csv` with pandas and plot: detections per hour, phone-on-desk time, average latency, confidence distribution.

### Step 10 — Stretch goals

- **Pan-tilt tracking** with 2 servos and a proportional controller: `e = x_object − x_center`, `u = Kp · e`.
- **MQTT** instead of HTTP responses (publish/subscribe, IoT standard).
- **Streamlit dashboard** showing the live annotated frame + charts.
- **ONNX / TFLite export** and latency benchmark.
- **Docker** container for the server.

## 6. Troubleshooting

| Problem | Likely cause / fix |
|---|---|
| `Camera init failed` | Bad ribbon connection, or PSRAM not enabled |
| Random reboots (brownout) | Power supply too weak — use 5 V / 2 A, short cable |
| Upload fails | GPIO 0 not connected to GND during flashing; press RESET after upload and remove the jumper |
| `HTTP error: -1` | Wrong server IP, firewall blocking port 8000, different network |
| Many false positives | Raise `conf`, increase debounce frames, add more training data |

## 7. Results (fill in with your own numbers)

| Model | mAP@0.5 | Precision | Recall | Latency (ms) |
|---|---|---|---|---|
| YOLOv8n pretrained | – | – | – | – |
| YOLOv8n custom | – | – | – | – |
| FOMO on ESP32 (edge) | – | – | – | – |

![demo](docs/demo.gif)

## 8. Roadmap

- [x] Camera streaming
- [x] Server-side YOLO inference
- [x] Actuator control (buzzer, relay)
- [ ] Custom dataset + training
- [ ] Edge inference comparison
- [ ] Pan-tilt tracking
- [ ] Dashboard

## 9. License

MIT — see `LICENSE`.

## 10. Author

**Vagif** — Automation Engineering @ ADNSU · aiming at Physical AI / AIoT.
[LinkedIn](#) · [GitHub](#)
