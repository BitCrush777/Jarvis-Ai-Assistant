# Jarvis Ai Assistant

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Platform: Windows](https://img.shields.io/badge/platform-Windows%2010%20%7C%2011-0078d4.svg)](https://www.microsoft.com/windows)
[![PyQt6 HUD](https://img.shields.io/badge/UI-PyQt6%20Cyberpunk%20HUD-00d4ff.svg)](https://riverbankcomputing.com/software/pyqt/)
[![MediaPipe](https://img.shields.io/badge/Vision-MediaPipe%20Tasks-FF6F00.svg)](https://developers.google.com/mediapipe)
[![NVIDIA NIM](https://img.shields.io/badge/AI-NVIDIA%20NIM%20%2F%20Ollama-76B900.svg)](https://build.nvidia.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An advanced, multimodal AI desktop assistant engineered for Windows. Combines ultra-low-latency voice interaction, NVIDIA NIM & local LLM reasoning, real-time computer vision, precision 720p HD hand-gesture control, multi-monitor window choreography, autonomous system automation, and WhatsApp Desktop voice bridges into a unified cyberpunk PyQt6 Heads-Up Display.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Technology Stack](#technology-stack)
- [AI & NVIDIA NIM Integration](#ai--nvidia-nim-integration)
- [Voice & Audio Pipeline](#voice--audio-pipeline)
- [Computer Vision & Screen Intelligence](#computer-vision--screen-intelligence)
- [High-Precision Hand Gesture Control](#high-precision-hand-gesture-control)
- [Window & Multi-Monitor Choreography](#window--multi-monitor-choreography)
- [System Automation & Tool Registry](#system-automation--tool-registry)
- [WhatsApp Desktop Automation & Voice Bridge](#whatsapp-desktop-automation--voice-bridge)
- [Autonomous Dev Agent & Coding](#autonomous-dev-agent--coding)
- [Persistent Memory System](#persistent-memory-system)
- [User Interface: PyQt6 HUD & Web Dashboard](#user-interface-pyqt6-hud--web-dashboard)
- [Plugin Architecture](#plugin-architecture)
- [Project Structure](#project-structure)
- [Prerequisites & Requirements](#prerequisites--requirements)
- [Installation & Quick Start](#installation--quick-start)
- [Configuration Guide](#configuration-guide)
- [Security & Privacy Audit](#security--privacy-audit)
- [Diagnostics & Verification](#diagnostics--verification)
- [Roadmap](#roadmap)
- [License](#license)

---

## Project Overview

**Jarvis Ai Assistant** is an extensible desktop AI agent designed to act as an autonomous operating system companion on Windows. Rather than existing merely as a text chatbot, JARVIS interfaces directly with physical audio devices, webcams, display drivers, window managers, running processes, local browser sessions, and external messaging channels.

The assistant is built around a single-instance core architecture, featuring thread-safe background workers, asynchronous pipelines, and decoupled UI event loops that ensure continuous responsiveness without dropping audio frames or stalling visual animations.

---

## Key Features

- **Multimodal Intelligence**: Cloud reasoning via **NVIDIA NIM** (`meta/llama-3.3-70b-instruct`, `nvidia/nemotron-3-super-120b-a12b`), Google Gemini Live API, or 100% offline local inference via **Ollama** (`llama3.2`).
- **Precision Hand-Gesture Tracking**: 720p HD real-time 21-joint 3D hand tracking powered by Google MediaPipe Tasks with `OneEuroFilter` sub-pixel jitter reduction, dynamic anti-aliased HUD skeleton overlays, and gesture-driven mouse control.
- **Natural Voice Conversations**: Real-time voice loop featuring continuous adaptive Voice Activity Detection (VAD), GPU-accelerated speech-to-text (`faster-whisper`), and natural neural text-to-speech (`EdgeTTS`).
- **Multi-Monitor Window Management**: Seamless window migration across arbitrary multi-monitor topologies with proportional geometry scaling, active window state tracking, and cycle navigation.
- **WhatsApp Desktop Voice Bridge**: Automated calling, WASAPI loopback audio capture, automated answering verification via Windows UI Automation (UIA), and full-duplex conversational audio bridging with self-echo cancellation.
- **Desktop Automation**: Application launching, volume control, media playback, system monitoring, file management, YouTube navigation, and browser control via Playwright.
- **Autonomous Dev Agent**: Self-directed coding assistant capable of codebase inspection, test execution, file modifications, and bug fixing.
- **Cyberpunk Arc-Reactor HUD**: Futuristic PyQt6 dark-theme interface with reactive sound wave visualizer, live camera viewport, gesture telemetry panel, and real-time system gauges.
- **Encrypted Web & Mobile Dashboard**: Fast local FastAPI web interface with QR-code mobile pairing and encrypted WebSocket telemetry for remote monitoring.
- **Hot-Pluggable Plugin System**: Extensible plugin directory (`plugins/`) with automated discovery, crash isolation, and dynamic tool schema registration.

---

## System Architecture

```text
                                  +---------------------------------------+
                                  |                 USER                  |
                                  |   (Voice, Gestures, HUD, Dashboard)   |
                                  +-------------------+-------------------+
                                                      |
                                                      v
+---------------------------------------------------------------------------------------------------------+
|                                          PERCEPTION SUBSYSTEM                                           |
|  +---------------------------+  +----------------------------+  +------------------------------------+  |
|  |       AUDIO CAPTURE       |  |       CAMERA SERVICE       |  |          SCREEN & VISION           |  |
|  | • Adaptive VAD (Energy)   |  | • DirectShow / YUY2 720p   |  | • Screen grabber (MSS)             |  |
|  | • Ring Buffer (60s)       |  | • MediaPipe 21 Landmarks   |  | • Multi-monitor bounding boxes    |  |
|  | • WASAPI Loopback         |  | • OneEuroFilter Smoothing  |  | • NVIDIA Vision NIM analysis       |  |
|  +-------------+-------------+  +-------------+--------------+  +-----------------+------------------+  |
+----------------|------------------------------|-----------------------------------|---------------------+
                 |                              |                                   |
                 v                              v                                   v
+---------------------------------------------------------------------------------------------------------+
|                                              JARVIS CORE                                                |
|  +---------------------------+  +----------------------------+  +------------------------------------+  |
|  |      SPEECH PIPELINE      |  |      GESTURE ENGINE        |  |          AI REASONING LAYER        |  |
|  | • faster-whisper (CUDA)   |  | • Static Pose Classifier   |  | • NVIDIA NIM API Router            |  |
|  | • EdgeTTS Neural Voice    |  | • Dynamic Swipe & Pinch    |  | • Ollama Local LLM Engine          |  |
|  | • Echo Suppression       |  | • Safe Action Dispatcher   |  | • Tool Use & Function Calling      |  |
|  +---------------------------+  +----------------------------+  +------------------------------------+  |
|                                                                                                         |
|  +---------------------------------------------------------------------------------------------------+  |
|  |                                      CONTEXT & MEMORY LAYER                                       |  |
|  | • Long-Term Memory (Identity, Preferences, Projects)  • Conversation History Window               |  |
|  +---------------------------------------------------------------------------------------------------+  |
+---------------------------------------------------+-----------------------------------------------------+
                                                    |
                                                    v
+---------------------------------------------------------------------------------------------------------+
|                                        EXECUTION & DISPATCH LAYER                                       |
|  +---------------------------+  +----------------------------+  +------------------------------------+  |
|  |     SYSTEM AUTOMATION     |  |       WINDOW MANAGER       |  |         COMMUNICATIONS & IO        |  |
|  | • Computer & App Control  |  | • Win32 Display Topology   |  | • WhatsApp Desktop UIA Bridge      |  |
|  | • Browser (Playwright)    |  | • Proportional Relocation  |  | • Mobile Encrypted WebSocket       |  |
|  | • Dev Agent & Code Tools  |  | • Gesture Window Switching |  | • Virtual Audio Cable Routing      |  |
|  +---------------------------+  +----------------------------+  +------------------------------------+  |
+---------------------------------------------------------------------------------------------------------+
                                                    |
                                                    v
+---------------------------------------------------------------------------------------------------------+
|                                        OUTPUT & PRESENTATION LAYER                                      |
|  +---------------------------------------------------------+  +--------------------------------------+  |
|  |                 PYQT6 HUD INTERFACE                     |  |         REMOTE WEB DASHBOARD         |  |
|  | • Live 720p Camera Preview  • Real-Time Audio Reactor   |  | • FastAPI HTTPS Server               |  |
|  | • Gesture Status Badges     • Active Window Telemetry   |  | • Mobile QR Pairing & Voice Bridge   |  |
|  +---------------------------------------------------------+  +--------------------------------------+  |
+---------------------------------------------------------------------------------------------------------+
```

---

## Technology Stack

| Domain | Technology / Library | Role |
|---|---|---|
| **Core & UI** | Python 3.11+, PyQt6, QtMultimedia | Application core, multi-threaded GUI, Cyberpunk HUD |
| **Cloud AI** | NVIDIA NIM API (`requests`, OpenAI-compatible) | High-speed cloud reasoning and multimodal visual analysis |
| **Local AI** | Ollama, CTranslate2, Google GenAI | Offline local LLMs (`llama3.2`, `qwen`), Gemini fallback |
| **Vision & Gestures** | OpenCV (`cv2`), MediaPipe Tasks, `pygrabber` | 720p HD capture, 21-point hand landmarking, DirectShow graph |
| **Audio & Speech** | `sounddevice`, `SoundCard`, `faster-whisper`, `EdgeTTS` | WASAPI loopback, adaptive VAD, CUDA-accelerated STT, neural TTS |
| **Automation** | `pywinauto`, `pywin32`, `uiautomation`, `Playwright` | Windows UI Automation, Win32 window APIs, browser execution |
| **Remote Dashboard** | FastAPI, Uvicorn, WebSockets, `cryptography` | Authenticated remote mobile dashboard, audio streaming |
| **Filtering & Math** | NumPy, SciPy, OneEuroFilter | Dynamic coordinate scaling, signal jitter filtering |

---

## AI & NVIDIA NIM Integration

JARVIS natively supports the **NVIDIA NIM (Inference Microservice)** cloud API platform, providing sub-second Time-to-First-Token (TTFT) and tool execution capabilities.

### Supported NVIDIA Models
- `meta/llama-3.3-70b-instruct` (Default reasoning & multi-turn dialog)
- `nvidia/nemotron-3-super-120b-a12b` (Deep analytical reasoning)
- `meta/llama-3.2-11b-vision-instruct` (Multimodal screen & image understanding)

### Configuration Instructions
To configure NVIDIA NIM:
1. Obtain an API key from [build.nvidia.com](https://build.nvidia.com/).
2. Set the key in `config/api_keys.json`:
   ```json
   {
       "nvidia_api_key": "YOUR_NVIDIA_API_KEY_HERE",
       "llm_provider": "nvidia",
       "llm_model": "meta/llama-3.3-70b-instruct",
       "llm_url": "https://integrate.api.nvidia.com/v1"
   }
   ```
*(Keys starting with `nvapi-` are automatically authenticated using standard Bearer authorization headers).*

### Hybrid Local Fallback (Ollama)
If internet connectivity is lost or local privacy is desired, set `"llm_provider": "ollama"` and `"llm_model": "llama3.2"`. JARVIS will seamlessly route queries to your local `http://localhost:11434` instance.

---

## Voice & Audio Pipeline

The voice subsystem delivers fluid conversations with real-time feedback:
1. **Adaptive Voice Activity Detection (VAD)**: Dynamic energy estimation tracks background ambient noise and triggers speech recognition only during intentional vocalizations.
2. **Speech-to-Text (STT)**: Powered by `faster-whisper` running on NVIDIA CUDA 12 (with automatic fallback to int8 CPU). Transcribes speech in under 200 ms.
3. **Text-to-Speech (TTS)**: High-fidelity natural voice streaming via `EdgeTTS` (using voices such as `en-US-GuyNeural`) and low-latency playback via `pygame` and `miniaudio`.
4. **Self-Echo Cancellation**: While JARVIS is speaking, audio capture is automatically gated to prevent the assistant from listening to and transcribing its own synthesized voice.

---

## Computer Vision & Screen Intelligence

- **Real-Time Screen Perception**: Grabs desktop regions using `mss` for high-FPS capture.
- **Visual Question Answering**: JARVIS can inspect your current screen, analyze active IDE code windows, inspect browser layouts, and debug graphical errors using NVIDIA Vision NIM.
- **Privacy Controls**: Screen capture is only activated upon explicit voice or gesture command, and zero image data is persisted to disk.

---

## High-Precision Hand Gesture Control

The hand gesture subsystem provides touchless mouse and window control:
- **Native 720p HD Video Negotiation**: Probes physical webcam hardware via DirectShow (`cv2.CAP_DSHOW`) and locks into native 1280x720 at 30 FPS using uncompressed `YUY2` format.
- **MediaPipe Tasks Architecture**: Evaluates 21 3D joint coordinates per hand with sub-millisecond tensor execution.
- **OneEuroFilter Jitter Reduction**: Dual-stage adaptive low-pass filter eliminates fingertip tremble during static hovering while dynamically adapting cutoff frequency during rapid motion to ensure zero latency.
- **Sub-Pixel Cursor Mapping**: Translates index finger coordinates in an active bounding box `[0.15, 0.85, 0.20, 0.80]` into high-precision multi-monitor cursor coordinates.
- **Anti-Accident State Machine**: Every gesture requires confirmation stability frames, neutral palm re-arming, and automatic action cooldown to prevent accidental trigger loops.
- **HUD Live Preview**: Live video feed rendered directly inside the HUD with anti-aliased dynamic skeleton overlays and real-time tracking badges.

### Gesture Vocabulary

| Gesture | Pose | Default Mapped Action |
|---|---|---|
| **POINT** | Index finger extended | Direct cursor control (pointing mode) |
| **PINCH** | Thumb + Index tip distance < 45mm | Left-click (tap) / Drag-and-drop (hold > 0.25s) |
| **SWIPE RIGHT** | Rapid palm transition (dx > 0.15) | Navigate to Next Window (`Alt+Tab`) |
| **SWIPE LEFT** | Rapid palm transition (dx < -0.15) | Navigate to Previous Window (`Alt+Shift+Tab`) |
| **FIST** | All fingers folded | Minimize Active Window |
| **TWO FINGER** | Index + Middle extended | Enter / Exit Multi-Monitor Window Management Mode |
| **THUMBS UP** | Thumb upright, fist closed | Confirm / Execute Action (`Enter`) |
| **THUMBS DOWN** | Thumb downward, fist closed | Cancel / Dismiss (`Escape`) |
| **OPEN PALM** | All fingers extended | Neutral State (re-arms gesture detector) |

---

## Window & Multi-Monitor Choreography

Controlled via `core/window_manager.py`:
- Detects virtual desktop geometry across all attached monitors via Win32 `EnumDisplayMonitors`.
- Migrates windows between monitors proportionally, recalculating window bounds so that scale and relative position are maintained across mismatched resolutions (e.g. 1920x1080 to 2560x1440).
- Safe fallbacks prevent moving windows into off-screen coordinates or taskbar boundaries.

---

## System Automation & Tool Registry

JARVIS includes a production-grade action suite in `actions/`:
- **`computer_control.py`**: Keyboard shortcuts, volume adjustment, media keys, screen locking, power telemetry.
- **`browser_control.py`**: Automated browser tasks, web searches via DuckDuckGo, page content extraction via Playwright.
- **`file_controller.py` & `file_processor.py`**: Safe file search, document reading, presentation creation, and trash routing via `send2trash`.
- **`system_monitor.py`**: Real-time CPU, RAM, GPU, battery, and disk telemetry.
- **`reminder.py`**: System toast notifications and countdown reminders.

---

## WhatsApp Desktop Automation & Voice Bridge

Implemented in `plugins/whatsapp_voice_bridge.py` and `plugins/whatsapp_desktop_call.py`:
- Uses Windows UI Automation (`uiautomation`) to interact directly with the native WhatsApp Desktop application without browser scraping or third-party webhooks.
- Detects call states (`CALLING`, `RINGING`, `CONNECTED`, `ENDED`) via UI tree queries.
- Routes synthesized voice into WhatsApp calls via a Virtual Audio Cable (`CABLE Output`) while capturing caller voice using WASAPI loopback capture.

---

## Autonomous Dev Agent & Coding

Located in `actions/dev_agent.py` and `actions/code_helper.py`:
- Inspects repository structure, reads source files, identifies syntax issues, and suggests targeted patches.
- Runs local unit tests and diagnostics, parsing failure traces and attempting self-repair cycles.

---

## Persistent Memory System

Located in `memory/`:
- **`memory_manager.py`**: Hierarchical memory system tracking user preferences, current projects, identity traits, and notes.
- Structured storage safely isolates user memory in `memory/long_term.json` (protected by `.gitignore`).
- New setups initialize from `memory/long_term.example.json`.

---

## User Interface: PyQt6 HUD & Web Dashboard

### Cyberpunk PyQt6 HUD (`ui.py`)
- High-tech dark neon interface (`#00d4ff` cyan accents).
- Dynamic audio reactor canvas that pulses in sync with microphone and TTS waveforms.
- Embedded live camera viewport with real-time MediaPipe overlay.
- Gesture status badges, window selector, telemetry readouts, and activity logs.

### Remote Web Dashboard (`dashboard/`)
- Lightweight FastAPI server providing a web-based companion interface.
- One-time QR code generation for secure pairing with mobile devices on the same local network.
- Encrypted WebSocket transport for telemetry streaming and remote voice input.

---

## Plugin Architecture

JARVIS features a zero-code plugin loader in `core/plugin_loader.py`:
- Drop any Python file into `plugins/`.
- JARVIS inspects the module on startup, registers exported tool methods, and exposes them to the LLM tool router.
- Plugin crashes are isolated and will not bring down the main assistant loop.
- See `plugins/_template.py` for authoring guidelines.

---

## Project Structure

```text
Jarvis-Ai-Assistant/
├── actions/                  # Core action handlers & automation tools
│   ├── browser_control.py    # Playwright browser automation
│   ├── computer_control.py   # OS volume, keyboard, and system control
│   ├── dev_agent.py          # Autonomous coding & test execution agent
│   ├── file_controller.py    # File system management & safe trash routing
│   ├── screen_processor.py   # Screen capture & visual analysis
│   └── system_monitor.py     # Hardware telemetry (CPU, RAM, GPU)
├── config/                   # Configuration templates & application assets
│   ├── api_keys.example.json # Safe configuration template with placeholders
│   └── jarvis.ico            # High-resolution application icon
├── core/                     # Foundational subsystems
│   ├── audio_capture.py      # WASAPI loopback & adaptive VAD
│   ├── camera.py             # DirectShow camera acquisition (720p HD YUY2)
│   ├── cursor_control.py     # Sub-pixel mouse mapping & OneEuroFilter
│   ├── gesture_control.py    # Anti-accident state machine & gesture mapping
│   ├── hand_tracker.py       # MediaPipe 21-landmark tracking & HUD overlay
│   ├── llm_client.py         # Multi-provider LLM client (NVIDIA NIM / Ollama)
│   ├── plugin_loader.py      # Hot-reloading plugin registry
│   ├── stt.py                # faster-whisper speech recognition
│   ├── tts.py                # EdgeTTS neural speech synthesizer
│   └── window_manager.py     # Multi-monitor display topology manager
├── dashboard/                # Remote web & mobile dashboard
│   ├── server.py             # FastAPI server & WebSocket endpoints
│   └── static/               # Web client assets (HTML5, JS, CSS)
├── docs/                     # Technical specifications & UIA trees
├── memory/                   # Long-term knowledge base
│   ├── config_manager.py     # Dynamic configuration persistence
│   ├── memory_manager.py     # User knowledge & preference store
│   └── long_term.example.json# Clean memory schema template
├── models/                   # Local ML models (auto-downloaded on launch)
├── plugins/                  # Extensible user plugins
│   ├── _template.py          # Boilerplate for new custom skills
│   ├── ky_ufo_drone.py       # KY-UFO drone flight controller
│   ├── whatsapp_desktop_call.py # Native WhatsApp call automation
│   └── whatsapp_voice_bridge.py # Full-duplex conversational voice bridge
├── tests/                    # Comprehensive unit test suite (127 tests)
│   ├── test_audio_capture.py
│   ├── test_cursor_control.py
│   ├── test_gesture_control.py
│   ├── test_gesture_ui.py
│   ├── test_hand_tracker.py
│   ├── test_nvidia_mode.py
│   └── test_window_manager.py
├── tools/                    # Hardware diagnostics & verification CLI
│   └── camera_gesture_diagnostic.py # 7-stage automated diagnostic suite
├── .env.example              # Environment variable template
├── .gitignore                # Security-hardened git ignore specification
├── LICENSE                   # MIT License
├── main.py                   # CLI entrypoint & async supervisor
├── requirements.txt          # Python dependency specifications
├── run.bat                   # Quick-start Windows launcher script
└── ui.py                     # Cyberpunk PyQt6 GUI & gesture viewport
```

---

## Prerequisites & Requirements

- **Operating System**: Windows 10 or Windows 11 (64-bit).
- **Python**: Python 3.11 recommended.
- **Hardware**:
  - Webcam supporting 1280x720 30 FPS.
  - Microphone and speakers/headphones.
  - (Optional) NVIDIA GPU with CUDA 12 support for accelerated Whisper STT.
  - (Optional for WhatsApp Bridge) [VB-Audio Virtual Cable](https://vb-audio.com/Cable/).

---

## Installation & Quick Start

### 1. Clone the Repository
```powershell
git clone https://github.com/BitCrush777/Jarvis-Ai-Assistant.git
cd Jarvis-Ai-Assistant
```

### 2. Create and Activate Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

*(Optional: For GPU-accelerated Whisper on NVIDIA hardware)*:
```powershell
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4. Configure Application Settings
Copy the safe template to create your local config:
```powershell
Copy-Item config\api_keys.example.json config\api_keys.json
```
Edit `config/api_keys.json` with your preferred editor and insert your NVIDIA API key or local Ollama settings.

### 5. Launch JARVIS
```powershell
python main.py
```
Or double-click `run.bat`.

---

## Configuration Guide

Configuration is managed via `config/api_keys.json`. Essential parameters:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `nvidia_api_key` | string | `""` | NVIDIA NIM API key (`nvapi-...`) |
| `llm_provider` | string | `"nvidia"` | Provider: `"nvidia"`, `"ollama"`, or `"gemini"` |
| `llm_model` | string | `"meta/llama-3.3-70b-instruct"` | Model identifier |
| `camera_resolution` | string | `"auto"` | `"auto"`, `[1280, 720]`, or `[640, 480]` |
| `target_fps` | int | `30` | Target camera acquisition framerate |
| `gesture_control.enabled` | bool | `true` | Enable or disable hand gesture recognition |
| `cursor_control.enabled` | bool | `true` | Enable or disable hand-driven cursor navigation |
| `tts_voice` | string | `"en-US-GuyNeural"` | Neural speech voice identifier |

---

## Security & Privacy Audit

JARVIS is built with strict privacy and credential isolation standards:
- **Zero Committed Secrets**: `.gitignore` strictly excludes all active configuration (`config/api_keys.json`), cryptographic certificates (`config/certs/`), `.env` files, and local user memory.
- **Local Sandbox Execution**: Actions are checked against a whitelist before dispatch. Potentially dangerous system commands (`shutdown`, `rmdir`, arbitrary PowerShell scripts) are strictly blocked by `core/gesture_control.py`.
- **Offline Capable**: Can operate with zero internet dependency using local Whisper models and Ollama LLMs.
- **Camera Indicator**: Live video preview makes it obvious when camera sensors are active.

---

## Diagnostics & Verification

JARVIS includes an automated diagnostic suite to verify all hardware interfaces without launching the full UI:

```powershell
python tools/camera_gesture_diagnostic.py --all-headless
```

This runs 7 automated subsystem benchmarks:
1. **Camera Enumeration**: Audits Windows camera privacy permissions and lists connected physical hardware.
2. **Raw Capture Test**: Measures native capture framerate and frame stability over 3.0s.
3. **MediaPipe Tracking Test**: Verifies 21-landmark tracking inference latency and accuracy.
4. **Gesture Recognition**: Tests static and dynamic gesture classification algorithms.
5. **Safe Action Dispatcher**: Validates security whitelist enforcement against dangerous shell commands.
6. **Cursor Controller**: Verifies `OneEuroFilter` jitter attenuation and screen coordinate mirroring.
7. **Multi-Monitor Topology**: Validates display bounds and primary monitor detection.

To run the complete test suite:
```powershell
python -m unittest discover -s tests
```
*(All 127 unit tests pass with zero errors).*

---

## Roadmap

- [x] NVIDIA NIM cloud integration.
- [x] Real 720p HD camera acquisition with anti-aliased HUD skeleton overlays.
- [x] Multi-monitor proportional window choreography.
- [x] WhatsApp Desktop UI Automation voice bridge.
- [ ] Multi-hand bimanual gesture controls (pinch-to-zoom, two-hand rotate).
- [ ] Offline local small vision model (Moondream2 / LLaVA) integration.
- [ ] Cross-platform Linux (Wayland / X11) display manager support.

---

## License

This project is licensed under the [MIT License](LICENSE) - see the `LICENSE` file for details.
