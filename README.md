# JARVIS (Mark II) — Autonomous AI Desktop Assistant

<div align="center">

```
     ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗     █████╗ ██╗
     ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝    ██╔══██╗██║
     ██║███████║██████╔╝██║   ██║██║███████╗    ███████║██║
██   ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║    ██╔══██║██║
╚█████╔╝██║  ██║██║  ██║ ╚████╔╝ ██║███████║    ██║  ██║██║
 ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝    ╚═╝  ╚═╝╚═╝
```

### *Just A Rather Very Intelligent System*
**Next-Generation Multimodal AI Desktop Assistant for Windows**

[![GitHub Stars](https://img.shields.io/github/stars/BitCrush777/Jarvis-Ai-Assistant?style=for-the-badge&logo=github&color=00d4ff)](https://github.com/BitCrush777/Jarvis-Ai-Assistant/stargazers)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Platform Windows](https://img.shields.io/badge/Platform-Windows%2010%20%7C%2011-0078D4?style=for-the-badge&logo=windows&logoColor=white)](https://www.microsoft.com/windows)
[![NVIDIA NIM](https://img.shields.io/badge/NVIDIA-NIM%20Accelerated-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://build.nvidia.com/)
[![MediaPipe Tasks](https://img.shields.io/badge/Vision-MediaPipe%20Tasks-FF6F00?style=for-the-badge&logo=google&logoColor=white)](https://developers.google.com/mediapipe)
[![Unit Tests](https://img.shields.io/badge/Unit%20Tests-127%2F127%20Passed-brightgreen?style=for-the-badge&logo=checkmarx)](tests/)
[![License MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

[Features](#-key-features) • [Architecture](#-system-architecture) • [AI Providers](#-ai--nvidia-nim-integration) • [Gestures](#-high-precision-hand-gesture-control) • [Quick Start](#-quick-start-guide) • [Configuration](#-configuration-reference) • [Limitations](#-known-limitations) • [Diagnostics](#-troubleshooting--diagnostics)

</div>

---

## 🌟 Executive Overview

**Jarvis Ai Assistant** (Mark II) is an autonomous, multimodal personal AI agent engineered specifically for Microsoft Windows. Moving beyond standard text-based chat windows, JARVIS is an operating-system-level companion that perceives your voice, tracks your hands in 3D space, observes your active displays, manages multi-monitor workflows, automates complex desktop tasks, and interfaces with external messaging channels like WhatsApp Desktop.

The system is architected around a single-instance core supervisor with decoupled worker threads:
- **Primary Cloud Reasoning**: Powered by **NVIDIA NIM (Inference Microservice)** cloud APIs (`meta/llama-3.3-70b-instruct`, `nvidia/nemotron-3-super-120b-a12b`) for streaming conversation and tool execution.
- **Local Offline Reasoning**: Built-in support for **Ollama** (`llama3.2`) for fully offline, private intelligence.
- **Optional / Legacy Cloud Fallback**: Google Gemini (`google-genai`) integration remains in the codebase for Gemini Live streaming audio and grounded web search if a `gemini_api_key` is provided, but is **not required** for normal operation (DuckDuckGo handles web search by default).
- **Perception & Control**: Real-time 21-joint 3D hand tracking via Google **MediaPipe Tasks**, low-latency **faster-whisper** CUDA speech recognition, and an adaptive **OneEuroFilter** touchless cursor controller.

---

## ⚡ Key Features

| Subsystem | Capabilities | Technology Stack |
|---|---|---|
| **🧠 Hybrid Intelligence** | Cloud reasoning via NVIDIA NIM (`meta/llama-3.3-70b-instruct`, `nemotron-3-super-120b`), local offline inference via Ollama (`llama3.2`), and optional Gemini Live fallback. | NVIDIA NIM, Ollama, Google GenAI |
| **🖐️ Touchless Gestures** | Negotiates high-resolution webcam capture (targeting native 720p HD @ 30 FPS via DirectShow), 21-joint 3D hand landmarks, `OneEuroFilter` jitter suppression, anti-aliased HUD skeleton rendering, and cursor navigation. | MediaPipe Tasks, OpenCV, DirectShow |
| **🎙️ Voice Conversation** | Full-duplex conversational voice loop with continuous adaptive energy VAD, GPU-accelerated speech-to-text (`faster-whisper`), neural speech synthesis (`EdgeTTS`), and hardware self-echo gating. | `faster-whisper` (CUDA 12 / CPU INT8), `EdgeTTS`, `sounddevice` |
| **🖥️ Multi-Monitor Hub** | Dynamic display topology scanning, cross-monitor window migration with proportional resolution scaling, and gesture window cycling (`Alt+Tab`). | Win32 APIs, `pywin32`, `pygetwindow` |
| **📞 WhatsApp Voice Bridge** | Automated contact calling via native Windows UI Automation (UIA), WASAPI loopback capture, and two-way voice bridging with Virtual Audio Cable. | `uiautomation`, WASAPI Loopback, VB-Audio Cable |
| **🤖 Autonomous Dev Agent** | Self-directed coding assistant capable of inspecting codebases, executing test suites, analyzing errors, and proposing targeted bug fixes. | Python AST, Subprocess Sandboxing |
| **💻 Cyberpunk HUD & Web** | PyQt6 dark neon arc-reactor interface with reactive audio waveforms, embedded live camera feed, and an encrypted FastAPI mobile web dashboard. | PyQt6, FastAPI, WebSockets, `cryptography` |

---

## 🏗️ System Architecture

```mermaid
graph TD
    User([User]) <--> Perception[Perception Subsystem]
    
    subgraph Perception Subsystem
        Mic[Microphone / Adaptive VAD]
        Cam[DirectShow Webcam\nTarget: 720p YUY2]
        Screen[MSS Screen Grabber]
    end
    
    Perception --> Core[JARVIS Core Supervisor]
    
    subgraph JARVIS Core
        STT[faster-whisper CUDA / CPU Engine]
        Tracker[MediaPipe Tasks 21-Landmark Tracker]
        Vision[NVIDIA Vision NIM / Screen Processor]
        Memory[Hierarchical Long-Term Memory]
        StateMachine[Anti-Accident Gesture State Machine]
    end
    
    Core --> Brain{AI Reasoning Layer}
    
    subgraph Brain [AI Reasoning Layer]
        NvidiaNIM[Primary: NVIDIA NIM Cloud API\nmeta/llama-3.3-70b\nnemotron-3-super-120b]
        LocalOllama[Offline: Local Ollama Engine\nllama3.2 / qwen]
        GeminiFallback[Optional: Google Gemini Live\nRequires gemini_api_key]
        Router[Tool & Action Router]
    end
    
    Brain --> Execution[Dispatch & Execution Layer]
    
    subgraph Execution [Dispatch & Execution Layer]
        WinManager[Multi-Monitor Window Manager]
        CursorCtrl[Sub-Pixel Cursor Controller]
        SystemActions[22 System & Browser Actions]
        WhatsApp[WhatsApp Desktop UIA Bridge]
    end
    
    Execution --> Output[Presentation & Telemetry Layer]
    
    subgraph Output [Presentation & Telemetry Layer]
        TTS[EdgeTTS Neural Voice Engine]
        HUD[PyQt6 Cyberpunk Arc-Reactor HUD]
        WebDash[Remote FastAPI Web Dashboard]
    end
    
    Output <--> User
```

---

## 🖥️ Cyberpunk Heads-Up Display (PyQt6)

The primary interface is a custom-engineered, dark-neon cyberpunk HUD rendered using hardware-accelerated **PyQt6** (`ui.py`):

```text
+---------------------------------------------------------------------------------------+
|  J.A.R.V.I.S.  ::  AUTONOMOUS DESKTOP INTELLIGENCE                       [ - ][ □ ][ X ] |
+---------------------------------------------------------------------------------------+
|  [ SYSTEM GAUGES ]       |                  [ ARC REACTOR ]                  |  [ GESTURE CONTROL ]   |
|  • CPU: 14% @ 3.8 GHz    |                 .---.     .---.                   |  • CAMERA: 1280x720    |
|  • RAM: 8.2 / 16.0 GB    |                /     \   /     \                  |  • STATUS: ONLINE      |
|  • GPU: RTX 3050 (32°C)  |               |   (•) | | (•)   |                 |  • SKELETON: 21 JOINTS |
|  • VAD: SPEECH DETECTED  |                \     /   \     /                  |  • POSE: POINTING      |
|  • LLM: NVIDIA NIM 70B   |                 '---'     '---'                   |  • FILTER: OneEuro     |
|                          |            [ REACTIVE AUDIO WAVE ]                |  +-------------------+ |
|  [ ACTIVE WINDOW ]       |      ~~/\~~/\__/\~~/\__/\~~/\~~/\__/\~~           |  | [LIVE PREVIEW]    | |
|  • Code Editor (Display 1|                                                   |  | • Cyan Reticle    | |
|  • Size: 1920x1080       |  "All systems operational, Sir.                   |  | • Fingertip Halos | |
|  • Target Monitor: 2     |   Awaiting your voice or gesture command."        |  +-------------------+ |
+---------------------------------------------------------------------------------------+
|  [TERMINAL TELEMETRY]                                                                 |
|  [14:02:10] [Gesture] Switched active focus to 'Google Chrome' (Display 2).           |
|  [14:02:15] [NVIDIA] Stream response received via meta/llama-3.3-70b-instruct.       |
|  [14:02:18] [Audio] Audio capture loopback active. Zero-echo suppression engaged.    |
+---------------------------------------------------------------------------------------+
```

---

## 🖐️ High-Precision Hand Gesture Control

The vision pipeline captures video via DirectShow (`cv2.CAP_DSHOW`), probing candidate modes (`1280x720`, `1920x1080`, `960x540`, `640x480`) and locking into native 720p HD where supported by physical hardware.

### Dual-Stage OneEuroFilter
To resolve the dilemma between hovering stability and fast-action responsiveness:
$$\text{Cutoff frequency: } f_c = f_{c,\text{min}} + \beta \cdot |\dot{x}|$$
- **At rest ($|\dot{x}| \to 0$)**: The filter applies an aggressive smoothing cutoff ($f_{c,\text{min}} = 1.2\text{ Hz}$), suppressing micro-tremor and sensor noise to produce a steady cursor.
- **During rapid motion ($|\dot{x}| \gg 0$)**: The filter dynamically increases the cutoff proportional to velocity ($\beta = 0.05$), following quick intentional gestures with low latency.

### Gesture Command Matrix

```text
   [ OPEN PALM ]          [ POINTING ]            [ PINCH ]              [ FIST ]
     Neutral                 Cursor                Left-Click /          Minimize
    (Re-Arms)               Control                 Drag Hold             Window
   
   🖐️                     👉                    🤏                    ✊
```

| Gesture | Biomechanical Pose Condition | Triggered System Action | Safety & Debounce |
|---|---|---|---|
| **POINT** | Index finger extended, remaining fingers curled | Sub-pixel cursor tracking across monitors | Clamped to active work box |
| **PINCH** | Distance(Thumb Tip, Index Tip) < 45 mm | Single Click (tap) / Drag-and-drop (hold > 0.25s) | Gated to pointing mode |
| **SWIPE RIGHT** | Rapid palm displacement ($\Delta x > +0.15$) | Navigate Next Application (`Alt + Tab`) | 0.6s cooldown debounce |
| **SWIPE LEFT** | Rapid palm displacement ($\Delta x < -0.15$) | Navigate Previous Application (`Alt + Shift + Tab`) | 0.6s cooldown debounce |
| **FIST** | All 5 fingertips curled toward palm | Minimize Active Window | Single-fire execution lock |
| **TWO FINGER** | Index + Middle extended, Ring/Pinky curled | Enter / Exit Multi-Monitor Mode | Confirmation hysteresis |
| **THUMBS UP** | Thumb extended vertically, fist curled | Confirm Dialog / Press `Enter` | Neutral re-arm required |
| **THUMBS DOWN**| Thumb pointing downward, fist curled | Cancel / Press `Escape` | Neutral re-arm required |
| **OPEN PALM** | All 5 fingers extended and spread | Neutral baseline state | Re-arms gesture detector |

---

## 🧠 AI & NVIDIA NIM Integration

JARVIS natively supports the **NVIDIA NIM (Inference Microservice)** cloud API platform, providing streaming text and function calling over an OpenAI-compatible interface.

### Supported NVIDIA Models
- **`meta/llama-3.3-70b-instruct`** — Production reasoning, conversation, and function calling.
- **`nvidia/nemotron-3-super-120b-a12b`** — High-complexity logical analysis and architecture planning.
- **`meta/llama-3.2-11b-vision-instruct`** — Multimodal desktop screenshot analysis and code debugging.

### Configuration
Edit `config/api_keys.json` with your credentials:

```json
{
    "nvidia_api_key": "YOUR_NVIDIA_API_KEY_HERE",
    "llm_provider": "nvidia",
    "llm_model": "meta/llama-3.3-70b-instruct",
    "llm_url": "https://integrate.api.nvidia.com/v1"
}
```

> [!NOTE]
> Any key beginning with `nvapi-` is automatically routed with standard `Bearer` authorization headers to NVIDIA NIM endpoints.

### Offline Fallback (Ollama)
When working offline, set `"llm_provider": "ollama"` and `"llm_model": "llama3.2"`. JARVIS will communicate directly with your local Ollama instance on `http://localhost:11434`.

### Legacy / Optional Cloud Provider (Google Gemini)
Code for Google Gemini (`google-genai`) remains present in `main.py`, `dashboard/server.py`, and `actions/web_search.py`. If a user configures `"gemini_api_key"`, JARVIS can utilize Gemini Live streaming voice and Gemini grounded search. If no Gemini key is provided, web searches automatically use DuckDuckGo (`ddgs`) and all reasoning uses NVIDIA NIM or Ollama.

---

## 🎙️ Voice & Audio Pipeline

1. **Adaptive Energy VAD**: Continuous ambient noise floor tracking. Automatically filters out background keyboard typing and ambient room noise while latching onto vocal pitch.
2. **Speech-to-Text (STT)**: Utilizes `faster-whisper` on NVIDIA CUDA 12 for low-latency local speech transcription. Automatically falls back to INT8 quantization on CPU if CUDA is unavailable.
3. **Neural TTS**: Synthesizes expressive, human-like voice responses via Microsoft `EdgeTTS` (`en-US-GuyNeural`) with low-latency streaming playback via `miniaudio` and `pygame`.
4. **Self-Echo Gating**: While JARVIS is speaking, audio capture is automatically muted to prevent the assistant from listening to its own synthesized voice.

---

## 🖥️ Window & Multi-Monitor Choreography

Controlled via `core/window_manager.py`:
- Detects virtual desktop geometry across all attached monitors via Win32 `EnumDisplayMonitors`.
- Migrates windows between monitors proportionally, recalculating window bounds so that relative scale and positioning are maintained across mismatched resolutions (e.g. moving a window from a 1080p display to a 1440p display).
- Safe boundary clamping prevents moving windows into off-screen coordinates or taskbar boundaries.

---

## 📞 WhatsApp Desktop Voice Bridge

Implemented in `plugins/whatsapp_voice_bridge.py`:
- Interacts directly with native Windows **WhatsApp Desktop** using UI Automation (`uiautomation`).
- Monitors incoming calls and verifies call states (`CONNECTING`, `CALLING`, `CONNECTED`, `ENDED`) without browser scraping.
- Injects synthesized voice into WhatsApp calls via a Virtual Audio Cable (`CABLE Output`) while capturing caller responses through WASAPI loopback audio.

---

## 🛠️ Action & Tool Registry

JARVIS ships with 22 modular action controllers in `actions/`:

```text
actions/
├── background_monitor.py   # Background health & process watcher
├── browser_control.py      # Playwright browser driver & web automation
├── code_helper.py          # Python syntax analysis & code generation
├── computer_control.py     # Windows volume, sleep, lock, keyboard shortcuts
├── computer_settings.py    # Display, network, and audio endpoint management
├── desktop.py              # Shortcut organizer & desktop layout management
├── dev_agent.py            # Autonomous software testing and repair agent
├── file_controller.py      # Safe file creation, reading, and send2trash routing
├── file_processor.py       # Document summarization (PDF, PPTX, TXT)
├── flight_finder.py        # Travel & flight pricing lookups
├── game_updater.py         # Automated game client patcher
├── open_app.py             # Application launcher with fuzzy name resolution
├── proactive.py            # Context-aware user suggestions
├── pushup_counter.py       # Vision-based fitness counter
├── reminder.py             # System toast notifications & timer alarms
├── screen_processor.py     # Desktop screenshot capture & visual analysis
├── send_message.py         # System notification message dispatcher
├── system_monitor.py       # Real-time CPU, RAM, GPU, battery, and disk telemetry
├── upload_video.py         # Media publishing automator
├── weather_report.py       # Atmospheric forecasting
├── web_search.py           # DuckDuckGo clean search (Gemini grounded fallback)
└── youtube_video.py        # YouTube search, playback, and transcript extraction
```

---

## 🚀 Quick Start Guide

### Prerequisites
- **Operating System**: Windows 10 or Windows 11 (64-bit).
- **Python**: Python 3.11 recommended.
- **Hardware**: Integrated or USB Webcam, Microphone, and Audio Output.
- *(Optional)*: NVIDIA GPU with CUDA 12 for hardware-accelerated Whisper STT.
- *(Optional for WhatsApp)*: [VB-Audio Virtual Cable](https://vb-audio.com/Cable/).

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

*(Optional: Enable CUDA 12 GPU acceleration for Whisper)*:
```powershell
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4. Create Local Configuration
Copy the safe configuration template:
```powershell
Copy-Item config\api_keys.example.json config\api_keys.json
```

Open `config/api_keys.json` and insert your NVIDIA API key or configure local Ollama settings:
```json
{
    "nvidia_api_key": "YOUR_NVIDIA_API_KEY_HERE",
    "llm_provider": "nvidia",
    "llm_model": "meta/llama-3.3-70b-instruct",
    "camera_resolution": "auto",
    "target_fps": 30
}
```

### 5. Launch JARVIS
```powershell
python main.py
```
*Or double-click `run.bat`.*

---

## ⚙️ Configuration Reference

All settings are managed via `config/api_keys.json` (untracked and protected by `.gitignore`):

| Key | Type | Default | Description |
|---|---|---|---|
| `nvidia_api_key` | `string` | `""` | NVIDIA NIM Cloud API key (`nvapi-...`) |
| `llm_provider` | `string` | `"nvidia"` | Model provider: `"nvidia"`, `"ollama"`, or `"gemini"` |
| `llm_model` | `string` | `"meta/llama-3.3-70b-instruct"` | Model identifier |
| `llm_url` | `string` | `"https://integrate.api.nvidia.com/v1"` | API base endpoint |
| `assistant_name` | `string` | `"JARVIS"` | Activation name |
| `user_name` | `string` | `"User"` | Preferred user honorific |
| `tts_engine` | `string` | `"edgetts"` | TTS engine: `"edgetts"` or `"pyttsx3"` |
| `tts_voice` | `string` | `"en-US-GuyNeural"` | Speech voice name |
| `whisper_model` | `string` | `"base"` | Whisper model size: `tiny`, `base`, `small`, `medium` |
| `camera_index` | `int` | `0` | DirectShow hardware camera device index |
| `camera_resolution` | `string` | `"auto"` | Resolution negotiation: `"auto"`, `[1280, 720]`, `[640, 480]` |
| `camera_mirror` | `bool` | `true` | Mirror camera feed horizontally |
| `target_fps` | `int` | `30` | Camera capture framerate target |
| `gesture_control.enabled` | `bool` | `true` | Master toggle for gesture recognition |
| `gesture_control.confidence_threshold` | `float` | `0.65` | Minimum classification confidence |
| `gesture_control.cooldown_seconds` | `float` | `0.6` | Cooldown period between action triggers |
| `cursor_control.enabled` | `bool` | `true` | Hand-driven cursor navigation toggle |
| `cursor_control.dead_zone_px` | `float` | `2.5` | Sub-pixel jitter deadzone |
| `cursor_control.min_cutoff` | `float` | `1.2` | OneEuroFilter minimum cutoff frequency |
| `cursor_control.beta` | `float` | `0.05` | OneEuroFilter velocity responsiveness slope |

---

## 🔒 Security & Privacy Audit

- **Zero Committed Secrets**: `.gitignore` strictly protects `config/api_keys.json`, SSL certificates (`config/certs/`), `.env` files, and local user memory (`memory/*.json`).
- **Sandbox Action Whitelist**: All gesture and voice commands pass through an execution whitelist before reaching Windows shell APIs. Dangerous shell commands (`format`, `rmdir`, `shutdown`, arbitrary PowerShell executions) are strictly blocked.
- **Local Data Sovereignty**: Voice recordings, camera frames, and screen captures are evaluated in-memory and are **never** persisted to disk or sent to external servers outside your configured LLM provider.

---

## ⚠️ Known Limitations

- **Operating System Dependency**: JARVIS is built specifically for Microsoft Windows 10/11. It relies on Win32 API display topology, DirectShow COM interfaces, WASAPI loopback capture, and Windows UI Automation.
- **WhatsApp Voice Bridge**: Requires **WhatsApp Desktop** for Windows to be installed and active, along with [VB-Audio Virtual Cable](https://vb-audio.com/Cable/) configured as the WhatsApp microphone for bidirectional audio injection.
- **Hardware-Dependent Vision & Capture**: The negotiated resolution (up to 720p HD) and achieved FPS depend on your physical webcam sensor capabilities and DirectShow driver support.
- **Whisper Real-Time Performance**: GPU acceleration requires an NVIDIA GPU with CUDA 12 drivers. Running larger Whisper models on CPU without CUDA may introduce noticeable transcription latency.
- **External Network Access**: Cloud reasoning requires a valid NVIDIA NIM API key (`nvapi-...`). Without an internet connection, you must run a local Ollama instance (`ollama serve`).

---

## 🩺 Troubleshooting & Diagnostics

JARVIS includes a standalone 7-stage automated diagnostic suite to audit hardware without launching the full GUI:

```powershell
python tools/camera_gesture_diagnostic.py --all-headless
```

### Sample Automated Diagnostic Output:
```text
======================================================================
  JARVIS DIAGNOSTIC: SUMMARY MATRIX (Sample Verification Run)
======================================================================
COMPONENT                      STATUS
--------------------------------------------------
Camera Device Enumerate        [PASS] (USB2.0 HD UVC WebCam @ 1280x720)
Test A: Raw Camera Capture     [PASS] (720p HD YUY2 @ target 30 FPS)
Test B: MediaPipe Hand Track   [PASS] (21 Landmarks, hardware-dependent)
Test C: Gesture Classification [PASS] (Static Poses & Swipes verified)
Test D: Safe Action Whitelist  [PASS] (Malicious commands blocked)
Test E: Cursor Controller      [PASS] (OneEuroFilter Jitter Suppressed)
Test F: Multi-Monitor Topology [PASS] (Active Display Geometry verified)
--------------------------------------------------
[PASS] ALL 7 DIAGNOSTIC TESTS PASSED SUCCESSFULLY.
```

### Running Unit Tests
To verify all 127 subsystem test cases:
```powershell
python -m unittest discover -s tests
```
*(All 127 tests pass locally with exit code 0).*

---

## 🗺️ Project Roadmap

- [x] **NVIDIA NIM Cloud Reasoning**: Low-latency multimodal LLM integration.
- [x] **High-Resolution Camera Acquisition**: DirectShow negotiation with anti-aliased HUD skeleton rendering.
- [x] **Touchless Cursor Navigation**: OneEuroFilter dual-stage jitter reduction.
- [x] **WhatsApp Voice Bridge**: Native UIA automation and full-duplex conversational audio.
- [x] **Multi-Monitor Choreography**: Display topology scanning and proportional window migration.
- [ ] **Bimanual Gestures**: Two-hand pinch-to-zoom, rotate, and 3D window manipulation.
- [ ] **Local Small Vision Model**: Embedded Moondream2 / LLaVA integration for offline screen reasoning.
- [ ] **Cross-Platform Linux Support**: Wayland / X11 display backend and PipeWire audio pipeline.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with passion by [Sai Darshan (BitCrush777)](https://github.com/BitCrush777)**

*Star ⭐ this repository if JARVIS makes your desktop experience feel like the future!*

</div>
