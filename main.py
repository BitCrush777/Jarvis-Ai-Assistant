import platform as _platform
import subprocess as _subprocess
import sys as _sys

# ── Make stdout/stderr UTF-8 tolerant ────────────────────────────────────────
# On non-UTF-8 Windows consoles (cp1254/cp1252/cp936...) any print() containing
# an emoji raises UnicodeEncodeError.  Several of those prints sit inside except
# handlers, so the handler itself would blow up and skip the recovery code that
# follows it — turning a recoverable error into a silent hang.  errors="replace"
# makes every print safe.
for _stream in (_sys.stdout, _sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass   # frozen builds may have no real stream attached

# ── Nuclear: force CREATE_NO_WINDOW on EVERY subprocess call on Windows ───────
# This patches Popen itself, so no per-file flag is needed anywhere.
if _platform.system() == "Windows":
    _OrigPopen = _subprocess.Popen

    class _Popen(_OrigPopen):
        def __init__(self, args, **kw):
            kw["creationflags"] = kw.get("creationflags", 0) | _subprocess.CREATE_NO_WINDOW
            kw.pop("startupinfo", None)   # drop any stale/shared STARTUPINFO
            super().__init__(args, **                       kw)

    _subprocess.Popen = _Popen

# ─────────────────────────────────────────────────────────────────────────────

import asyncio
import queue as _queue
import re
import threading
import time
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

import sounddevice as sd
try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None
    types = None

if types is None:
    class _TypesFallback:
        class FunctionResponse:
            def __init__(self, name: str, response: dict, id: str = ""):
                self.name = name
                self.response = response
                self.id = id
    types = _TypesFallback()
from ui import JarvisUI
from memory.memory_manager import (
    load_memory, update_memory, format_memory_for_prompt,
    save_session_summary, pop_last_session,
)

from actions.file_processor import file_processor
from actions.flight_finder     import flight_finder
from actions.open_app          import open_app
from actions.weather_report    import weather_action
from actions.send_message      import send_message
from actions.reminder          import reminder
from actions.computer_settings import computer_settings
from actions.screen_processor  import _capture_camera, _capture_screen
from actions.youtube_video     import youtube_video
from actions.desktop           import desktop_control
from actions.browser_control   import browser_control
from actions.file_controller   import file_controller
from actions.code_helper       import code_helper
from actions.dev_agent         import dev_agent
from actions.web_search        import web_search as web_search_action
from actions.computer_control  import computer_control
from actions.game_updater      import game_updater
from actions.system_monitor    import SystemMonitor, get_system_status
from actions.proactive         import ProactiveEngine
from actions.background_monitor import (
    add_monitor, remove_monitor, list_monitors, check_all as monitor_check_all,
)
from actions.web_search        import _news as _fetch_news_sync
from memory.config_manager     import get_brief_enabled
from core.plugin_loader        import discover_plugins
from core.audio_capture        import AudioInputPipeline, enumerate_microphones, run_standalone_mic_test

def get_base_dir():
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent

BASE_DIR        = get_base_dir()
API_CONFIG_PATH = BASE_DIR / "config" / "api_keys.json"
PROMPT_PATH     = BASE_DIR / "core" / "prompt.txt"
LIVE_MODEL          = "models/gemini-2.5-flash-native-audio-preview-12-2025"
CHANNELS            = 1
SEND_SAMPLE_RATE    = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE          = 1024

def _get_api_key() -> str:
    try:
        with open(API_CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f).get("gemini_api_key", "")
    except Exception:
        return ""


def _read_config() -> dict:
    try:
        with open(API_CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _is_nvidia_mode() -> bool:
    try:
        from memory.config_manager import get_active_llm_provider, get_nvidia_key
        provider = get_active_llm_provider()
        if provider == "gemini":
            return False
        if provider == "nvidia":
            return True
        nvidia_k = get_nvidia_key()
        if nvidia_k and len(nvidia_k) > 10:
            return True
    except Exception:
        pass
    return True


def _clean_param_schema(schema):
    if isinstance(schema, dict):
        res = {}
        for k, v in schema.items():
            if k == "type" and isinstance(v, str):
                res[k] = v.lower()
            else:
                res[k] = _clean_param_schema(v)
        return res
    elif isinstance(schema, list):
        return [_clean_param_schema(item) for item in schema]
    return schema


def _to_openai_tools(gemini_tools: list) -> list:
    openai_tools = []
    for tool in gemini_tools:
        name = tool.get("name", "")
        desc = tool.get("description", "")
        params = tool.get("parameters", {})
        cleaned_params = _clean_param_schema(params)
        if isinstance(cleaned_params, dict) and "type" not in cleaned_params:
            cleaned_params["type"] = "object"
        openai_tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": desc,
                "parameters": cleaned_params,
            },
        })
    return openai_tools


class ToolCallWrapper:
    def __init__(self, name: str, args: dict, id: str = ""):
        self.name = name
        self.args = args
        self.id = id


def _load_system_prompt() -> str:
    try:
        return PROMPT_PATH.read_text(encoding="utf-8")
    except Exception:
        return (
            "You are JARVIS, Tony Stark's AI assistant. "
            "Be concise, direct, and always use the provided tools to complete tasks. "
            "Never simulate or guess results — always call the appropriate tool."
        )

_CTRL_RE = re.compile(r"<ctrl\d+>", re.IGNORECASE)

def _clean_transcript(text: str) -> str:    
    text = _CTRL_RE.sub("", text)
    text = re.sub(r"[\x00-\x08\x0b-\x1f]", "", text)
    return text.strip()

TOOL_DECLARATIONS = [
    {
        "name": "open_app",
        "description": (
            "Opens any application on the computer. "
            "Use this whenever the user asks to open, launch, or start any app, "
            "website, or program. Always call this tool — never just say you opened it."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "app_name": {
                    "type": "STRING",
                    "description": "Exact name of the application (e.g. 'WhatsApp', 'Chrome', 'Spotify')"
                }
            },
            "required": ["app_name"]
        }
    },
    {
        "name": "web_search",
        "description": (
            "Searches the web. Use for ANY question about current facts, events, prices, "
            "or topics — always prefer this over guessing. "
            "Modes: 'search' (default), 'news' (latest headlines on a topic), "
            "'research' (deep comprehensive answer), 'price' (product cost lookup), "
            "'compare' (side-by-side comparison of items)."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "query":  {"type": "STRING", "description": "Search query or topic"},
                "mode":   {"type": "STRING", "description": "search | news | research | price | compare"},
                "items":  {"type": "ARRAY",  "items": {"type": "STRING"}, "description": "Items to compare (compare mode)"},
                "aspect": {"type": "STRING", "description": "Comparison aspect: price | specs | reviews | features"},
            },
            "required": ["query"]
        }
    },
    {
        "name": "system_status",
        "description": (
            "Returns real-time system metrics: CPU usage, RAM, GPU load, CPU temperature, "
            "uptime, and process count. Use when the user asks about computer performance, "
            "temperature, memory, or resource usage."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {},
        }
    },
    {
        "name": "weather_report",
        "description": "Gives the weather report to user",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "city": {"type": "STRING", "description": "City name"}
            },
            "required": ["city"]
        }
    },
    {
        "name": "send_message",
        "description": "Sends a text message via WhatsApp, Telegram, or other messaging platform.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "receiver":     {"type": "STRING", "description": "Recipient contact name"},
                "message_text": {"type": "STRING", "description": "The message to send"},
                "platform":     {"type": "STRING", "description": "Platform: WhatsApp, Telegram, etc."}
            },
            "required": ["receiver", "message_text", "platform"]
        }
    },
    {
        "name": "reminder",
        "description": "Sets a timed reminder using Task Scheduler.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "date":    {"type": "STRING", "description": "Date in YYYY-MM-DD format"},
                "time":    {"type": "STRING", "description": "Time in HH:MM format (24h)"},
                "message": {"type": "STRING", "description": "Reminder message text"}
            },
            "required": ["date", "time", "message"]
        }
    },
    {
        "name": "youtube_video",
        "description": (
            "Controls YouTube. Use for: playing videos, summarizing a video's content, "
            "getting video info, or showing trending videos."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "play | summarize | get_info | trending (default: play)"},
                "query":  {"type": "STRING", "description": "Search query for play action"},
                "save":   {"type": "BOOLEAN", "description": "Save summary to Notepad (summarize only)"},
                "region": {"type": "STRING", "description": "Country code for trending e.g. TR, US"},
                "url":    {"type": "STRING", "description": "Video URL for get_info action"},
            },
            "required": []
        }
    },
    {
        "name": "screen_process",
        "description": (
            "Captures the screen or webcam image and lets you analyze it. "
            "MUST be called when user asks what is on screen, what you see, "
            "look at camera, analyze my screen, etc. "
            "You have NO visual ability without this tool. "
            "After the image is captured it is sent directly to you — describe what you see and answer the user's question. "
            "When using camera: the live view stays open until user says close it or calls close_camera."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "angle": {"type": "STRING", "description": "'screen' to capture display, 'camera' for webcam. Default: 'screen'"},
                "text":  {"type": "STRING", "description": "The question or instruction about the captured image"}
            },
            "required": ["text"]
        }
    },
    {
        "name": "close_camera",
        "description": (
            "Closes the live camera view shown on screen. "
            "Call when user says: close camera, stop camera, turn off camera, "
            "kamerayı kapat, kapat, creepy, etc."
        ),
        "parameters": {"type": "OBJECT", "properties": {}, "required": []}
    },
    {
        "name": "computer_settings",
        "description": (
            "Controls the computer: volume, brightness, window management, keyboard shortcuts, "
            "typing text on screen, closing apps, fullscreen, dark mode, WiFi, restart, shutdown, "
            "scrolling, tab management, zoom, screenshots, lock screen, refresh/reload page. "
            "Use for ANY single computer control command."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":      {"type": "STRING", "description": "The action to perform"},
                "description": {"type": "STRING", "description": "Natural language description of what to do"},
                "value":       {"type": "STRING", "description": "Optional value: volume level, text to type, etc."}
            },
            "required": []
        }
    },
    {
        "name": "browser_control",
        "description": (
            "Controls any web browser. Use for: opening websites, searching the web, "
            "clicking elements, filling forms, scrolling, screenshots, navigation, any web-based task. "
            "Simple open/search requests launch the user's own browser normally (their real profile "
            "and logged-in accounts); interactive actions (click, type, fill_form...) attach an "
            "automation browser. "
            "Always pass the 'browser' parameter when the user specifies a browser (e.g. 'open in Edge', "
            "'use Firefox', 'open Chrome'). Multiple browsers can run simultaneously."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":      {"type": "STRING", "description": "go_to | search | click | type | scroll | fill_form | smart_click | smart_type | get_text | get_url | press | new_tab | close_tab | screenshot | back | forward | reload | switch | list_browsers | close | close_all"},
                "browser":     {"type": "STRING", "description": "Target browser: chrome | edge | firefox | opera | operagx | brave | vivaldi | safari. Omit to use the currently active browser."},
                "url":         {"type": "STRING", "description": "URL for go_to / new_tab action"},
                "query":       {"type": "STRING", "description": "Search query for search action"},
                "engine":      {"type": "STRING", "description": "Search engine: google | bing | duckduckgo | yandex (default: google)"},
                "selector":    {"type": "STRING", "description": "CSS selector for click/type"},
                "text":        {"type": "STRING", "description": "Text to click or type"},
                "description": {"type": "STRING", "description": "Element description for smart_click/smart_type"},
                "direction":   {"type": "STRING", "description": "up | down for scroll"},
                "amount":      {"type": "INTEGER", "description": "Scroll amount in pixels (default: 500)"},
                "key":         {"type": "STRING", "description": "Key name for press action (e.g. Enter, Escape, F5)"},
                "path":        {"type": "STRING", "description": "Save path for screenshot"},
                "incognito":   {"type": "BOOLEAN", "description": "Open in private/incognito mode"},
                "clear_first": {"type": "BOOLEAN", "description": "Clear field before typing (default: true)"},
            },
            "required": ["action"]
        }
    },
    {
        "name": "file_controller",
        "description": "Manages files and folders: list, create, delete, move, copy, rename, read, write, find, disk usage.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":      {"type": "STRING", "description": "list | create_file | create_folder | delete | move | copy | rename | read | write | find | largest | disk_usage | organize_desktop | info"},
                "path":        {"type": "STRING", "description": "File/folder path or shortcut: desktop, downloads, documents, home"},
                "destination": {"type": "STRING", "description": "Destination path for move/copy"},
                "new_name":    {"type": "STRING", "description": "New name for rename"},
                "content":     {"type": "STRING", "description": "Content for create_file/write"},
                "name":        {"type": "STRING", "description": "File name to search for"},
                "extension":   {"type": "STRING", "description": "File extension to search (e.g. .pdf)"},
                "count":       {"type": "INTEGER", "description": "Number of results for largest"},
            },
            "required": ["action"]
        }
    },
    {
        "name": "desktop_control",
        "description": "Controls the desktop: wallpaper, organize, clean, list, stats.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "wallpaper | wallpaper_url | organize | clean | list | stats | task"},
                "path":   {"type": "STRING", "description": "Image path for wallpaper"},
                "url":    {"type": "STRING", "description": "Image URL for wallpaper_url"},
                "mode":   {"type": "STRING", "description": "by_type or by_date for organize"},
                "task":   {"type": "STRING", "description": "Natural language desktop task"},
            },
            "required": ["action"]
        }
    },
    {
        "name": "code_helper",
        "description": "Writes, edits, explains, runs, or builds code files.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":      {"type": "STRING", "description": "write | edit | explain | run | build | auto (default: auto)"},
                "description": {"type": "STRING", "description": "What the code should do or what change to make"},
                "language":    {"type": "STRING", "description": "Programming language (default: python)"},
                "output_path": {"type": "STRING", "description": "Where to save the file"},
                "file_path":   {"type": "STRING", "description": "Path to existing file for edit/explain/run/build"},
                "code":        {"type": "STRING", "description": "Raw code string for explain"},
                "args":        {"type": "STRING", "description": "CLI arguments for run/build"},
                "timeout":     {"type": "INTEGER", "description": "Execution timeout in seconds (default: 30)"},
            },
            "required": ["action"]
        }
    },
    {
        "name": "dev_agent",
        "description": "Builds complete multi-file projects from scratch: plans, writes files, installs deps, opens VSCode, runs and fixes errors.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "description":  {"type": "STRING", "description": "What the project should do"},
                "language":     {"type": "STRING", "description": "Programming language (default: python)"},
                "project_name": {"type": "STRING", "description": "Optional project folder name"},
                "timeout":      {"type": "INTEGER", "description": "Run timeout in seconds (default: 30)"},
            },
            "required": ["description"]
        }
    },
    {
        "name": "computer_control",
        "description": "Direct computer control: type, click, hotkeys, scroll, move mouse, screenshots, find elements on screen.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":      {"type": "STRING", "description": "type | smart_type | click | double_click | right_click | hotkey | press | scroll | move | copy | paste | screenshot | wait | clear_field | focus_window | screen_find | screen_click | random_data | user_data"},
                "text":        {"type": "STRING", "description": "Text to type or paste"},
                "x":           {"type": "INTEGER", "description": "X coordinate"},
                "y":           {"type": "INTEGER", "description": "Y coordinate"},
                "keys":        {"type": "STRING", "description": "Key combination e.g. 'ctrl+c'"},
                "key":         {"type": "STRING", "description": "Single key e.g. 'enter'"},
                "direction":   {"type": "STRING", "description": "up | down | left | right"},
                "amount":      {"type": "INTEGER", "description": "Scroll amount (default: 3)"},
                "seconds":     {"type": "NUMBER",  "description": "Seconds to wait"},
                "title":       {"type": "STRING",  "description": "Window title for focus_window"},
                "description": {"type": "STRING",  "description": "Element description for screen_find/screen_click"},
                "type":        {"type": "STRING",  "description": "Data type for random_data"},
                "field":       {"type": "STRING",  "description": "Field for user_data: name|email|city"},
                "clear_first": {"type": "BOOLEAN", "description": "Clear field before typing (default: true)"},
                "path":        {"type": "STRING",  "description": "Save path for screenshot"},
            },
            "required": ["action"]
        }
    },
    {
        "name": "game_updater",
        "description": (
            "THE ONLY tool for ANY Steam or Epic Games request. "
            "Use for: installing, downloading, updating games, listing installed games, "
            "checking download status, scheduling updates. "
            "ALWAYS call directly for any Steam/Epic/game request. "
            "NEVER use browser_control or web_search for Steam/Epic."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":    {"type": "STRING",  "description": "update | install | list | download_status | schedule | cancel_schedule | schedule_status (default: update)"},
                "platform":  {"type": "STRING",  "description": "steam | epic | both (default: both)"},
                "game_name": {"type": "STRING",  "description": "Game name (partial match supported)"},
                "app_id":    {"type": "STRING",  "description": "Steam AppID for install (optional)"},
                "hour":      {"type": "INTEGER", "description": "Hour for scheduled update 0-23 (default: 3)"},
                "minute":    {"type": "INTEGER", "description": "Minute for scheduled update 0-59 (default: 0)"},
                "shutdown_when_done": {"type": "BOOLEAN", "description": "Shut down PC when download finishes"},
            },
            "required": []
        }
    },
    {
        "name": "flight_finder",
        "description": "Searches Google Flights and speaks the best options.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "origin":      {"type": "STRING",  "description": "Departure city or airport code"},
                "destination": {"type": "STRING",  "description": "Arrival city or airport code"},
                "date":        {"type": "STRING",  "description": "Departure date (any format)"},
                "return_date": {"type": "STRING",  "description": "Return date for round trips"},
                "passengers":  {"type": "INTEGER", "description": "Number of passengers (default: 1)"},
                "cabin":       {"type": "STRING",  "description": "economy | premium | business | first"},
                "save":        {"type": "BOOLEAN", "description": "Save results to Notepad"},
            },
            "required": ["origin", "destination", "date"]
        }
    },
    {
        "name": "manage_monitor",
        "description": (
            "Add, remove, or list background monitoring topics. "
            "JARVIS checks these topics once a day and alerts the user when there is a new development. "
            "Use 'add' when the user says 'monitor X', 'track X', 'follow X'. "
            "Use 'remove' when the user says 'stop monitoring X'. "
            "Use 'list' when the user asks what is being monitored. "
            "Do NOT add crypto, financial, or trading topics."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {
                    "type":        "STRING",
                    "description": "add | remove | list",
                },
                "topic": {
                    "type":        "STRING",
                    "description": "Topic to monitor or stop monitoring (e.g. 'space exploration', 'AI news')",
                },
            },
            "required": ["action"],
        },
    },
    {
        "name": "shutdown_jarvis",
        "description": (
            "Shuts down the assistant completely. "
            "Call this when the user expresses intent to end the conversation, "
            "close the assistant, say goodbye, or stop Jarvis. "
            "The user can say this in ANY language."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {},
        }
    },
    {
    "name": "file_processor",
    "description": (
        "Processes any file that the user has uploaded or dropped onto the interface. "
        "Use this when the user refers to an uploaded file and wants an action on it. "
        "Supports: images (describe/ocr/resize/compress/convert), "
        "PDFs (summarize/extract_text/to_word), "
        "Word docs & text files (summarize/fix/reformat/translate), "
        "CSV/Excel (analyze/stats/filter/sort/convert), "
        "JSON/XML (validate/format/analyze), "
        "code files (explain/review/fix/optimize/run/document/test), "
        "audio (transcribe/trim/convert/info), "
        "video (trim/extract_audio/extract_frame/compress/transcribe/info), "
        "archives (list/extract), "
        "presentations (summarize/extract_text). "
        "ALWAYS call this tool when a file has been uploaded and the user gives a command about it. "
        "If the user's command is ambiguous, pick the most logical action for that file type."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "file_path": {
                "type": "STRING",
                "description": "Full path to the uploaded file. Leave empty to use the currently uploaded file."
            },
            "action": {
                "type": "STRING",
                "description": (
                    "What to do with the file. Examples by type:\n"
                    "image: describe | ocr | resize | compress | convert | info\n"
                    "pdf: summarize | extract_text | to_word | info\n"
                    "docx/txt: summarize | fix | reformat | translate_hint | word_count | to_bullet\n"
                    "csv/excel: analyze | stats | filter | sort | convert | info\n"
                    "json: validate | format | analyze | to_csv\n"
                    "code: explain | review | fix | optimize | run | document | test\n"
                    "audio: transcribe | trim | convert | info\n"
                    "video: trim | extract_audio | extract_frame | compress | transcribe | info | convert\n"
                    "archive: list | extract\n"
                    "pptx: summarize | extract_text | analyze"
                )
            },
            "instruction": {
                "type": "STRING",
                "description": "Free-form instruction if action doesn't cover it. E.g. 'translate this to Turkish', 'find all email addresses'"
            },
            "format": {
                "type": "STRING",
                "description": "Target format for conversion. E.g. 'mp3', 'pdf', 'csv', 'png'"
            },
            "width":     {"type": "INTEGER", "description": "Target width for image resize"},
            "height":    {"type": "INTEGER", "description": "Target height for image resize"},
            "scale":     {"type": "NUMBER",  "description": "Scale factor for image resize (e.g. 0.5)"},
            "quality":   {"type": "INTEGER", "description": "Quality 1-100 for image/video compress"},
            "start":     {"type": "STRING",  "description": "Start time for trim: seconds or HH:MM:SS"},
            "end":       {"type": "STRING",  "description": "End time for trim: seconds or HH:MM:SS"},
            "timestamp": {"type": "STRING",  "description": "Timestamp for video frame extraction HH:MM:SS"},
            "column":    {"type": "STRING",  "description": "Column name for CSV filter/sort"},
            "value":     {"type": "STRING",  "description": "Filter value for CSV filter"},
            "condition": {"type": "STRING",  "description": "Filter condition: equals|contains|gt|lt"},
            "ascending": {"type": "BOOLEAN", "description": "Sort order for CSV sort (default: true)"},
            "save":      {"type": "BOOLEAN", "description": "Save result to file (default: true)"},
            "destination": {"type": "STRING", "description": "Output folder for archive extract"},
        },
        "required": []
    }
},
    {
        "name": "save_memory",
        "description": (
            "Save an important personal fact about the user to long-term memory. "
            "Call this silently whenever the user reveals something worth remembering: "
            "name, age, city, job, preferences, hobbies, relationships, projects, or future plans. "
            "Do NOT call for: weather, reminders, searches, or one-time commands. "
            "Do NOT announce that you are saving — just call it silently. "
            "Values must be in English regardless of the conversation language."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "category": {
                    "type": "STRING",
                    "description": (
                        "identity — name, age, birthday, city, job, language, nationality | "
                        "preferences — favorite food/color/music/film/game/sport, hobbies | "
                        "projects — active projects, goals, things being built | "
                        "relationships — friends, family, partner, colleagues | "
                        "wishes — future plans, things to buy, travel dreams | "
                        "notes — habits, schedule, anything else worth remembering"
                    )
                },
                "key":   {"type": "STRING", "description": "Short snake_case key (e.g. name, favorite_food, sister_name)"},
                "value": {"type": "STRING", "description": "Concise value in English (e.g. Saidarshan, pizza, older sister)"},
            },
            "required": ["category", "key", "value"]
        }
    },
]

class JarvisLive:

    def __init__(self, ui: JarvisUI):
        self.ui             = ui
        self._asst_name     = "JARVIS"   # updated each session from config
        self._user_name     = ""
        self.session              = None
        self.audio_in_queue       = None
        self.out_queue            = None
        self._loop                = None
        self._is_speaking         = False
        self._speaking_lock       = threading.Lock()
        self._phone_active        = False   # True while phone mic is streaming; pauses PC mic
        self._pending_vision       = None    # (img_bytes, mime_type, question, angle) to inject after tool response
        self._vision_cam_active    = False   # True if camera was opened for vision → auto-close after response
        self._vision_close_pending = False   # True after vision injected; next turn_complete closes camera
        self._vision_last_time     = 0.0     # monotonic time of last screen_process call (cooldown guard)
        self._vision_busy          = False   # True while a vision capture/inject cycle is in flight
        self._interrupted          = False   # True while draining audio after user interrupt
        self._tts_player           = None
        self._audio_pipeline       = None
        self._nvidia_text_queue: asyncio.Queue | None = None
        self.ui.on_text_command   = self._on_text_command
        self.ui.on_remote_clicked = self._make_remote_key
        self.ui.on_interrupt      = self.interrupt
        self.ui.on_mute_change    = self._on_mute_change
        self._turn_done_event: asyncio.Event | None = None
        self._dashboard     = None
        self._briefing_sent    = False          # morning briefing fires once per process
        self._sys_monitor      = SystemMonitor()  # persistent cooldown state
        self._proactive        = ProactiveEngine()
        self._last_user_speech = time.monotonic()  # updated on every user utterance
        self._session_log: list[str] = []          # conversation turns for end-of-session summary

        self._enhanced_live = True  # affective dialog + proactive audio; auto-disabled if the server rejects them
        _core_names = {t["name"] for t in TOOL_DECLARATIONS}
        self._plugin_registry = discover_plugins(
            plugins_dir=Path(__file__).resolve().parent / "plugins",
            core_tool_names=_core_names,
            logger=lambda msg: (print(f"[Plugins] {msg}"), self.ui.write_log(f"SYS: {msg}")),
        )
        self.ui.get_plugins = self._plugin_registry.list_for_ui
        self.ui.request_say = self.plugin_say   # plugins: mid-task speech channel

    def plugin_say(self, instruction: str) -> None:
        """
        Thread-safe speech channel for plugins: lets a plugin ask JARVIS to
        say something short WHILE its run() is still executing (plugins block
        their executor thread, so they can't speak through the tool response
        until they finish). The instruction is injected into the Live session
        exactly like a proactive check-in; Gemini phrases it naturally in the
        user's language. Silently a no-op when no session is connected.
        """
        if _is_nvidia_mode():
            self.speak(instruction)
            return

        loop = getattr(self, "_loop", None)
        if not loop or not self.session:
            return

        async def _say():
            try:
                await self.session.send_client_content(
                    turns={"parts": [{"text": instruction}]},
                    turn_complete=True,
                )
            except Exception as e:
                print(f"[PluginSay] {e}")

        try:
            asyncio.run_coroutine_threadsafe(_say(), loop)
        except Exception as e:
            print(f"[PluginSay] {e}")

    def _make_remote_key(self):
        """Called from Qt main thread when user presses Remote Control."""
        if self._dashboard is None:
            self.ui.write_log(
                "SYS: Dashboard unavailable. "
                "Run: pip install fastapi \"uvicorn[standard]\" cryptography"
            )
            return None
        key    = self._dashboard.new_key()
        url    = self._dashboard.get_url()
        manual = self._dashboard.get_manual_url()
        return url, key, f"{url}/auto-login?key={key}", manual

    def _on_text_command(self, text: str):
        if _is_nvidia_mode():
            if self._nvidia_text_queue and self._loop:
                self._loop.call_soon_threadsafe(self._nvidia_text_queue.put_nowait, text)
            return
        if not self._loop or not self.session:
            return
        asyncio.run_coroutine_threadsafe(
            self.session.send_client_content(
                turns={"parts": [{"text": text}]},
                turn_complete=True
            ),
            self._loop
        )

    def set_speaking(self, value: bool):
        with self._speaking_lock:
            self._is_speaking = value
        if value:
            self.ui.set_state("SPEAKING")
        elif not self.ui.muted:
            self.ui.set_state("LISTENING")

    def _on_mute_change(self, muted: bool) -> None:
        if self._audio_pipeline:
            if muted:
                self._audio_pipeline.pause()
            else:
                self._audio_pipeline.resume()

    def interrupt(self) -> None:
        """Stop JARVIS mid-speech: drain queued audio and open mic immediately."""
        self._interrupted = True
        if self._tts_player:
            try:
                self._tts_player.stop()
            except Exception:
                pass
        if self._audio_pipeline and not getattr(self.ui, "muted", False):
            self._audio_pipeline.resume()
        q = self.audio_in_queue
        if q:
            drained = 0
            while True:
                try:
                    q.get_nowait()
                    drained += 1
                except Exception:
                    break
            if drained:
                print(f"[JARVIS] ✋ Interrupted — {drained} audio chunks discarded")
        self.set_speaking(False)
        if self._turn_done_event:
            self._turn_done_event.clear()
        self.ui.write_log("SYS: Interrupted — listening...")

    def speak(self, text: str):
        if _is_nvidia_mode():
            if self._tts_player:
                def _do_speak():
                    self.set_speaking(True)
                    if self._audio_pipeline:
                        self._audio_pipeline.pause()
                    try:
                        self._tts_player.speak(text)
                    finally:
                        self.set_speaking(False)
                        time.sleep(0.35)
                        if self._audio_pipeline and not getattr(self.ui, "muted", False):
                            self._audio_pipeline.resume()
                threading.Thread(target=_do_speak, daemon=True).start()
            return
        if not self._loop or not self.session:
            return
        asyncio.run_coroutine_threadsafe(
            self.session.send_client_content(
                turns={"parts": [{"text": text}]},
                turn_complete=True
            ),
            self._loop
        )

    def speak_error(self, tool_name: str, error: str):
        short = str(error)[:120]
        self.ui.write_log(f"ERR: {tool_name} — {short}")
        self.speak(f"Sir, {tool_name} encountered an error. {short}")

    _TOOL_KEYWORDS = (
        # Applications & processes
        "open ", "launch ", "start ", "close ", "kill ", "quit ", "exit ",
        # Web & search
        "search ", "google ", "look up ", "lookup ", "browse ", "latest news", "price of", "compare ",
        # Weather
        "weather", "forecast", "temperature", "rain",
        # System status
        "cpu", "gpu", "ram usage", "memory usage", "battery", "disk usage", "system status",
        # Media & volume
        "volume", "mute", "unmute", "play ", "pause", "stop music", "resume", "skip", "spotify", "youtube",
        # WhatsApp & messaging
        "whatsapp", "message", "text to", "call ",
        # Drone
        "drone", "takeoff", "land the", "fly ",
        # Screen & vision & control
        "screenshot", "screen", "what's on my screen", "what am i looking at",
        "camera", "webcam", "click", "type ", "press ", "scroll",
        # Files
        "file", "folder", "directory", "pdf", "convert ", "compress ",
        # Memory & reminders
        "remind me", "set a reminder", "set an alarm", "set a timer", "remember that", "remember this",
    )

    def _needs_tools(self, text: str) -> bool:
        """
        Checks if the user's input likely requires tool execution.
        If False, tool schemas can be omitted, reducing prompt tokens by ~95% and slashing TTFT.
        """
        t = f" {text.lower().strip()} "
        for kw in self._TOOL_KEYWORDS:
            if kw in t:
                return True
        return False

    def _check_fast_path(self, text: str) -> str | None:
        """
        Evaluates immediate conversational queries locally in <1ms without calling cloud LLM.
        Returns the response string, or None if the query should go to LLM.
        """
        t = text.lower().strip().rstrip(".?!")

        # Strip leading assistant invocation: "jarvis, what is my name" -> "what is my name"
        clean_asst = (self._asst_name or "jarvis").lower().strip()
        if clean_asst and t.startswith(clean_asst):
            t = t[len(clean_asst):].strip().lstrip(",:;- ").strip()

        # Guard: do not intercept multi-part or action queries
        conjunctions = (" and ", " then ", " but ", " after ", " also ", " or ")
        if any(c in t for c in conjunctions):
            return None

        # 1. User identity
        name_patterns = (
            "what is my name", "what's my name", "who am i", "do you know my name",
            "tell me my name", "my name"
        )
        if t in name_patterns:
            if self._user_name:
                return f"Your name is {self._user_name}, Sir."
            return "You have not told me your name yet, Sir."

        # 2. Assistant identity
        asst_patterns = (
            "what is your name", "what's your name", "who are you",
            "what are you called", "tell me your name", "identify yourself"
        )
        if t in asst_patterns:
            return f"I am {self._asst_name}, your desktop assistant, Sir."

        # 3. Current time
        time_patterns = (
            "what time is it", "what is the time", "what's the time",
            "tell me the time", "current time", "time please", "the time"
        )
        if t in time_patterns:
            now_str = datetime.now().strftime("%I:%M %p")
            return f"It is currently {now_str}, Sir."

        # 4. Current date
        date_patterns = (
            "what is today's date", "what's today's date", "what is the date",
            "what's the date", "today's date", "what day is it", "what day is today"
        )
        if t in date_patterns:
            date_str = datetime.now().strftime("%A, %B %d, %Y")
            return f"Today is {date_str}, Sir."

        # 5. Status / Health
        status_patterns = (
            "how are you", "how are you doing", "how're you",
            "are you there", "you there", "status report", "system status check"
        )
        if t in status_patterns:
            return f"All systems are operational and running at peak performance, Sir."

        return None

    def _build_system_instruction_text(self) -> str:
        from datetime import datetime

        # Load customization from config
        try:
            _cfg = json.loads(open(API_CONFIG_PATH, encoding="utf-8").read())
            self._asst_name = (_cfg.get("assistant_name") or "JARVIS").strip()
            self._user_name = (_cfg.get("user_name") or "").strip()
            _user_name = self._user_name
        except Exception:
            self._asst_name = "JARVIS"
            self._user_name = ""
            _user_name = ""

        memory     = load_memory()
        mem_str    = format_memory_for_prompt(memory)
        sys_prompt = _load_system_prompt()

        now      = datetime.now()
        time_str = now.strftime("%A, %B %d, %Y — %I:%M %p")
        time_ctx = (
            f"[CURRENT DATE & TIME]\n"
            f"Right now it is: {time_str}\n"
            f"Use this to calculate exact times for reminders.\n\n"
        )

        # Identity injection — overrides any hardcoded name in prompt.txt
        _addr = (f"ADDRESS: Always call the user '{_user_name}'."
                 if _user_name
                 else "ADDRESS: When speaking Turkish → always say \"efendim\". "
                      "When speaking English → say \"sir\". Never mix languages.")
        identity_ctx = (
            f"[IDENTITY]\n"
            f"Your name is {self._asst_name}. "
            f"Always refer to yourself as {self._asst_name}.\n"
            f"{_addr}\n\n"
        )

        parts = [time_ctx, identity_ctx]
        if mem_str:
            parts.append(mem_str)
        parts.append(sys_prompt)
        return "\n".join(parts)

    def _build_config(self) -> types.LiveConnectConfig:
        instruction = self._build_system_instruction_text()
        cfg = dict(
            response_modalities=["AUDIO"],
            output_audio_transcription={},
            input_audio_transcription={},
            system_instruction=instruction,
            tools=[{"function_declarations": TOOL_DECLARATIONS + self._plugin_registry.get_tool_declarations()}],
            session_resumption=types.SessionResumptionConfig(),
            # Sliding-window compression: session never dies from a full context
            # window — JARVIS can stay in one conversation for hours
            context_window_compression=types.ContextWindowCompressionConfig(
                sliding_window=types.SlidingWindow(),
            ),
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name="Charon"
                    )
                )
            ),
        )
        if self._enhanced_live:
            # Affective dialog: JARVIS hears tone/emotion and adapts its voice.
            # Proactive audio: JARVIS stays silent when speech isn't addressed
            # to it (background chatter, talking to someone else in the room).
            cfg["enable_affective_dialog"] = True
            cfg["proactivity"] = types.ProactivityConfig(proactive_audio=True)
        return types.LiveConnectConfig(**cfg)

    async def _execute_tool(self, fc) -> types.FunctionResponse:
        name = fc.name
        args = dict(fc.args or {})

        print(f"[JARVIS] 🔧 {name}  {args}")
        self.ui.set_state("THINKING")

        if name == "save_memory":
            category = args.get("category", "notes")
            key      = args.get("key", "")
            value    = args.get("value", "")
            if key and value:
                update_memory({category: {key: {"value": value}}})
                print(f"[Memory] 💾 save_memory: {category}/{key} = {value}")
            if not self.ui.muted:
                self.ui.set_state("LISTENING")
            return types.FunctionResponse(
                id=fc.id, name=name,
                response={"result": "ok", "silent": True}
            )

        loop   = asyncio.get_event_loop()
        result = "Done."

        try:
            if name == "open_app":
                r = await loop.run_in_executor(None, lambda: open_app(parameters=args, response=None, player=self.ui))
                result = r or f"Opened {args.get('app_name')}."

            elif name == "weather_report":
                r = await loop.run_in_executor(None, lambda: weather_action(parameters=args, player=self.ui))
                result = r or "Weather delivered."

            elif name == "browser_control":
                r = await loop.run_in_executor(None, lambda: browser_control(parameters=args, player=self.ui))
                result = r or "Done."

            elif name == "file_controller":
                r = await loop.run_in_executor(None, lambda: file_controller(parameters=args, player=self.ui))
                result = r or "Done."

            elif name == "send_message":
                r = await loop.run_in_executor(None, lambda: send_message(parameters=args, response=None, player=self.ui, session_memory=None))
                result = r or f"Message sent to {args.get('receiver')}."

            elif name == "reminder":
                r = await loop.run_in_executor(None, lambda: reminder(parameters=args, response=None, player=self.ui))
                result = r or "Reminder set."

            elif name == "youtube_video":
                r = await loop.run_in_executor(None, lambda: youtube_video(parameters=args, response=None, player=self.ui))
                result = r or "Done."

            elif name == "screen_process":
                import time as _t_mod
                _now = _t_mod.monotonic()
                _cooldown = 4.0  # seconds — covers echo window after speaking ends
                if self._vision_busy or (_now - self._vision_last_time) < _cooldown:
                    _wait = max(0, _cooldown - (_now - self._vision_last_time))
                    print(f"[Vision] ⏳ Cooldown active ({_wait:.1f}s remaining) — ignoring duplicate call")
                    result = "Vision is still processing the previous request. I will not call this again."
                else:
                    self._vision_busy      = True
                    self._vision_last_time = _now
                    angle     = args.get("angle", "screen").lower()
                    user_text = args.get("text", "What do you see?")
                    if angle == "camera":
                        img_b, mime_t = await loop.run_in_executor(None, _capture_camera)
                        self.ui.start_camera_stream()
                        self._vision_cam_active = True
                        print(f"[Vision] 📷 Camera: {len(img_b):,} bytes")
                        _stall = "camera"
                    else:
                        img_b, mime_t = await loop.run_in_executor(None, _capture_screen)
                        print(f"[Vision] 🖥️  Screen: {len(img_b):,} bytes")
                        _stall = "screen"
                    self._pending_vision = (img_b, mime_t, user_text, angle)
                    result = (
                        f"[VISION_ACTIVE] {_stall.capitalize()} captured. "
                        f"Immediately say ONE short natural sentence in the user's own language, "
                        f"telling them you are looking at their {_stall} right now. "
                        f"Do NOT describe or guess content — the actual image arrives in the NEXT message."
                    )

            elif name == "close_camera":
                self.ui.stop_camera_stream()
                result = "Camera closed."

            elif name == "computer_settings":
                r = await loop.run_in_executor(None, lambda: computer_settings(parameters=args, response=None, player=self.ui))
                result = r or "Done."

            elif name == "desktop_control":
                r = await loop.run_in_executor(None, lambda: desktop_control(parameters=args, player=self.ui))
                result = r or "Done."

            elif name == "code_helper":
                r = await loop.run_in_executor(None, lambda: code_helper(parameters=args, player=self.ui, speak=self.speak))
                result = r or "Done."

            elif name == "dev_agent":
                r = await loop.run_in_executor(None, lambda: dev_agent(parameters=args, player=self.ui, speak=self.speak))
                result = r or "Done."

            elif name == "web_search":
                r = await loop.run_in_executor(None, lambda: web_search_action(parameters=args, player=self.ui))
                result = r or "Done."
                # Mirror results to the on-screen content panel
                _mode = args.get("mode", "search")
                if r and not r.startswith("No results") and not r.startswith("Search failed"):
                    _query = args.get("query") or ", ".join(args.get("items", []))
                    _label = f"{_mode.upper()} — {_query[:38]}" if _query else _mode.upper()
                    self.ui.show_content(_label, r)
            elif name == "file_processor":
                if not args.get("file_path") and self.ui.current_file:
                    args["file_path"] = self.ui.current_file
                r = await loop.run_in_executor(
                    None,
                    lambda: file_processor(parameters=args, player=self.ui, speak=self.speak)
                )
                result = r or "Done."

            elif name == "computer_control":
                r = await loop.run_in_executor(None, lambda: computer_control(parameters=args, player=self.ui))
                result = r or "Done."

            elif name == "game_updater":
                r = await loop.run_in_executor(None, lambda: game_updater(parameters=args, player=self.ui, speak=self.speak))
                result = r or "Done."

            elif name == "flight_finder":
                r = await loop.run_in_executor(None, lambda: flight_finder(parameters=args, player=self.ui))
                result = r or "Done."

            elif name == "system_status":
                r = await loop.run_in_executor(None, get_system_status)
                result = str(r)

            elif name == "manage_monitor":
                action = args.get("action", "").lower().strip()
                topic  = args.get("topic", "").strip()
                if action == "add" and topic:
                    result = await asyncio.to_thread(add_monitor, topic)
                elif action == "remove" and topic:
                    result = await asyncio.to_thread(remove_monitor, topic)
                elif action == "list":
                    topics = await asyncio.to_thread(list_monitors)
                    result = ("Monitoring: " + ", ".join(topics)) if topics else "No topics are being monitored."
                else:
                    result = "Specify action (add/remove/list) and a topic."

            elif name == "shutdown_jarvis":
                self.ui.write_log("SYS: Shutdown requested.")
                async def _do_shutdown():
                    await self._save_session_summary()
                    if self.session:
                        try:
                            await self.session.send_client_content(
                                turns={"parts": [{"text": "Say a brief natural goodbye to the user."}]},
                                turn_complete=True,
                            )
                        except Exception:
                            pass
                    await asyncio.sleep(1.5)
                    import os as _os
                    _os._exit(0)
                asyncio.create_task(_do_shutdown())

            else:
                if self._plugin_registry.has(name):
                    r = await loop.run_in_executor(
                        None,
                        lambda: self._plugin_registry.run(name, args, player=self.ui, session_memory=None)
                    )
                    result = r or "Done."
                else:
                    result = f"Unknown tool: {name}"

        except Exception as e:
            result = f"Tool '{name}' failed: {e}"
            traceback.print_exc()
            self.speak_error(name, e)

        if not self.ui.muted:
            self.ui.set_state("LISTENING")

        print(f"[JARVIS] 📤 {name} → {str(result)[:80]}")
        return types.FunctionResponse(
            id=fc.id, name=name,
            response={"result": result}
        )

    async def _send_realtime(self):
        while True:
            msg = await self.out_queue.get()
            await self.session.send_realtime_input(media=msg)

    async def _listen_audio(self):
        print("[JARVIS] 🎤 Mic started")
        loop = asyncio.get_event_loop()

        def callback(indata, frames, time_info, status):
            with self._speaking_lock:
                jarvis_speaking = self._is_speaking
            if not jarvis_speaking and not self.ui.muted and not self._phone_active:
                data = indata.tobytes()
                loop.call_soon_threadsafe(
                    self.out_queue.put_nowait,
                    {"data": data, "mime_type": "audio/pcm"}
                )

        try:
            with sd.InputStream(
                samplerate=SEND_SAMPLE_RATE,
                channels=CHANNELS,
                dtype="int16",
                blocksize=CHUNK_SIZE,
                callback=callback,
            ):
                print("[JARVIS] 🎤 Mic stream open")
                while True:
                    await asyncio.sleep(0.1)
        except Exception as e:
            print(f"[JARVIS] ❌ Mic: {e}")
            raise

    async def _receive_audio(self):
        print("[JARVIS] 👂 Recv started")
        out_buf, in_buf = [], []

        try:
            while True:
                async for response in self.session.receive():

                    if response.data:
                        if self._interrupted:
                            pass  # discard: interrupted
                        else:
                            if self._turn_done_event and self._turn_done_event.is_set():
                                self._turn_done_event.clear()
                            # Split into ~50 ms chunks so interrupt() stops audio within 50 ms
                            # (24000 Hz × 2 bytes/sample × 0.05 s = 2400 bytes per slice)
                            _audio_data = response.data
                            _SLICE = 2400
                            for _i in range(0, len(_audio_data), _SLICE):
                                self.audio_in_queue.put_nowait(_audio_data[_i : _i + _SLICE])

                    if response.server_content:
                        sc = response.server_content

                        if sc.output_transcription and sc.output_transcription.text:
                            txt = _clean_transcript(sc.output_transcription.text)
                            if txt and txt != (out_buf[-1] if out_buf else ""):
                                out_buf.append(txt)

                        if sc.input_transcription and sc.input_transcription.text:
                            txt = _clean_transcript(sc.input_transcription.text)
                            if txt:
                                in_buf.append(txt)
                                self._last_user_speech = time.monotonic()

                        if sc.turn_complete:
                            if self._turn_done_event:
                                self._turn_done_event.set()

                            # If this turn_complete ends an interrupted response, clear the
                            # flag and skip all further processing for that turn.
                            if self._interrupted:
                                self._interrupted = False
                                in_buf  = []
                                out_buf = []
                                continue

                            full_in = " ".join(in_buf).strip()
                            if full_in:
                                self.ui.write_log(f"You: {full_in}")
                                self._session_log.append(f"User: {full_in}")
                                if self._dashboard:
                                    asyncio.create_task(self._dashboard.broadcast({
                                        "type": "log", "speaker": "user",
                                        "text": full_in,
                                        "ts": datetime.now().isoformat(),
                                    }))
                            in_buf = []

                            full_out = " ".join(out_buf).strip()
                            if full_out:
                                self.ui.write_log(f"{self._asst_name}: {full_out}")
                                self._session_log.append(f"{self._asst_name}: {full_out}")
                                if self._dashboard:
                                    asyncio.create_task(self._dashboard.broadcast({
                                        "type": "log", "speaker": "jarvis",
                                        "text": full_out,
                                        "ts": datetime.now().isoformat(),
                                    }))
                            out_buf = []

                            # Vision injection: model finished tool-response turn → now send the image
                            if self._pending_vision and self.session:
                                import base64 as _b64
                                img_b, mime_t, question, angle = self._pending_vision
                                self._pending_vision = None
                                b64 = _b64.b64encode(img_b).decode("ascii")
                                print(f"[Vision] 📤 {len(img_b):,} bytes (angle={angle}) → main session")
                                await self.session.send_client_content(
                                    turns={"parts": [
                                        {"inline_data": {"mime_type": mime_t, "data": b64}},
                                        {"text": question},
                                    ]},
                                    turn_complete=True,
                                )
                                # Mark next turn_complete behaviour depending on angle
                                if self._vision_cam_active:
                                    # Camera: keep busy until JARVIS finishes speaking the answer
                                    self._vision_cam_active    = False
                                    self._vision_close_pending = True
                                else:
                                    # Screen-only: no camera to close; release busy flag now
                                    self._vision_busy = False
                            elif self._vision_close_pending:
                                # This turn_complete IS the vision answer — close camera + release busy flag
                                self._vision_close_pending = False
                                self._vision_busy = False
                                async def _cam_close():
                                    await asyncio.sleep(2.0)
                                    self.ui.stop_camera_stream()
                                asyncio.create_task(_cam_close())

                    if response.tool_call:
                        fn_responses = []
                        for fc in response.tool_call.function_calls:
                            print(f"[JARVIS] 📞 {fc.name}")
                            fr = await self._execute_tool(fc)
                            fn_responses.append(fr)
                        await self.session.send_tool_response(
                            function_responses=fn_responses
                        )
        except Exception as e:
            print(f"[JARVIS] ❌ Recv: {e}")
            traceback.print_exc()
            raise

    async def _play_audio(self):
        print("[JARVIS] 🔊 Play started")

        stream = sd.RawOutputStream(
            samplerate=RECEIVE_SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            blocksize=CHUNK_SIZE,
        )
        stream.start()

        try:
            while True:
                try:
                    chunk = await asyncio.wait_for(
                        self.audio_in_queue.get(),
                        timeout=0.1
                    )
                except asyncio.TimeoutError:
                    if (
                        self._turn_done_event
                        and self._turn_done_event.is_set()
                        and self.audio_in_queue.empty()
                    ):
                        self.set_speaking(False)
                        self._turn_done_event.clear()
                    continue

                self.set_speaking(True)

                # Batch all immediately-available chunks into one write to reduce
                # thread-pool round-trips (was one asyncio.to_thread per 50ms slice).
                # Cap at ~200 ms so interrupt() still stops audio within ~200 ms.
                batch = bytearray(chunk)
                while len(batch) < 9600:   # 9600 bytes ≈ 200 ms at 24 kHz / 16-bit mono
                    try:
                        batch.extend(self.audio_in_queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break

                try:
                    await asyncio.to_thread(stream.write, bytes(batch))
                except (RuntimeError, asyncio.CancelledError):
                    break   # executor shutting down — exit cleanly
        except Exception as e:
            print(f"[JARVIS] ❌ Play: {e}")
            raise
        finally:
            self.set_speaking(False)
            stream.stop()
            stream.close()

    # ── Morning briefing ────────────────────────────────────────────────────────

    async def _inject_prompt(self, text: str) -> None:
        if _is_nvidia_mode():
            if self._nvidia_text_queue:
                await self._nvidia_text_queue.put(text)
            return
        if self.session:
            await self.session.send_client_content(
                turns={"parts": [{"text": text}]},
                turn_complete=True,
            )

    async def _send_startup_briefing(self) -> None:
        """
        Two-phase briefing optimized for speed:
          Phase 1 — instant greeting (no tools) → speech starts in <1s
          Phase 2 — news pre-fetched in a background thread while Phase 1 plays,
                    delivered as ready text (no Gemini tool-call round-trip) and
                    shown on the UI content panel. Waits for turn_complete event
                    instead of a fixed sleep so there is no unnecessary gap.
        """
        memory   = load_memory()
        identity = memory.get("identity", {})

        def _val(k: str) -> str:
            e = identity.get(k, {})
            return (e.get("value", "") if isinstance(e, dict) else str(e)).strip()

        lang = _val("language")
        name = _val("name")
        time_str = datetime.now().strftime("%H:%M")

        # Start fetching news immediately — runs in parallel while phase 1 plays
        loop = asyncio.get_event_loop()
        news_future = loop.run_in_executor(None, _fetch_news_sync, "top world news today")

        await asyncio.sleep(0.3)
        if not self.session and not _is_nvidia_mode():
            return

        # ── Phase 1: instant greeting ─────────────────────────────────────────
        lang_clause = f" Respond in {lang}." if lang else ""
        name_clause = f" Address the user as {name}." if name else ""

        # Inject last session context if available — pop removes it so it's never repeated
        last = await asyncio.to_thread(pop_last_session)
        session_clause = ""
        if last:
            try:
                _delta = (datetime.now() - datetime.strptime(last["date"], "%Y-%m-%d")).days
                _when  = "earlier today" if _delta == 0 else ("yesterday" if _delta == 1 else f"{_delta} days ago")
            except Exception:
                _when = "last time"
            session_clause = (
                f" Also briefly and naturally mention that {_when}: {last['summary']}"
            )

        p1 = (
            f"Greet the user warmly, mention it is {time_str}, and say you are fetching today's news now.{session_clause} "
            f"Keep it to 2 short sentences max. Do not call any tools.{lang_clause}{name_clause}"
        )

        # Clear the turn-done event so we can wait for Phase 1 to finish
        if self._turn_done_event:
            self._turn_done_event.clear()

        await self._inject_prompt(p1)
        self.ui.write_log("SYS: Briefing phase 1 (greeting) sent.")

        # ── Phase 2: fire as soon as Phase 1 audio is done ───────────────────
        async def _deliver_news():
            try:
                lang_str = f" Respond in {lang}." if lang else ""

                # Wait for news fetch (already running) and Phase 1 turn-complete
                # in parallel — whichever takes longer determines the wait time
                news_done   = asyncio.wrap_future(news_future)
                turn_waited = False
                if self._turn_done_event:
                    try:
                        await asyncio.wait_for(self._turn_done_event.wait(), timeout=6.0)
                        turn_waited = True
                    except asyncio.TimeoutError:
                        pass

                # Extra buffer: turn_complete fires when Gemini finishes *generating*
                # Phase 1, but audio may still be playing.  Waiting a beat here
                # prevents Phase 2 audio from arriving while Phase 1 is mid-sentence
                # (which sounds like a "repeated first response" to the user).
                if turn_waited:
                    await asyncio.sleep(0.8)
                else:
                    await asyncio.sleep(1.0)

                try:
                    news_text = await asyncio.wait_for(news_done, timeout=8.0)
                except Exception as e:
                    self.ui.write_log(f"SYS: News fetch timed out/failed: {e!r}")
                    news_text = ""

                if not self.session and not _is_nvidia_mode():
                    return

                failed = (not news_text) or news_text.startswith(
                    ("No news found", "Search failed", "Please provide")
                )
                if not failed:
                    # Show on UI content panel immediately
                    self.ui.show_content("NEWS — top world news today", news_text)

                    p2 = (
                        f"[BRIEFING] Here are today's top news headlines:\n{news_text}\n\n"
                        "Pick ONE headline, summarise it in one sentence, then say the full list "
                        f"is displayed on screen. Do not call any tools.{lang_str}"
                    )
                else:
                    self.ui.write_log(
                        f"SYS: News unavailable — backend returned: {news_text[:120]!r}"
                    )
                    p2 = (
                        "News headlines could not be fetched right now. "
                        f"Let the user know briefly.{lang_str}"
                    )

                await self._inject_prompt(p2)
                self.ui.write_log("SYS: Briefing phase 2 (news) sent.")
            except Exception as e:
                print(f"[Briefing] Phase 2 error: {e}")
                self.ui.write_log(f"SYS: Briefing phase 2 failed: {e}")

        asyncio.create_task(_deliver_news())

    # ── Session memory ──────────────────────────────────────────────────────────

    async def _save_session_summary(self) -> None:
        """Summarise the current session in 1-2 sentences and save to long_term.json."""
        log = self._session_log
        if len(log) < 3:          # need at least one exchange to be worth saving
            return
        self._session_log = []    # reset immediately so the next session starts clean

        memory = load_memory()
        lang_entry = memory.get("identity", {}).get("language", {})
        lang = (lang_entry.get("value", "") if isinstance(lang_entry, dict) else str(lang_entry)).strip()
        lang = lang or "English"

        convo = "\n".join(log[-40:])   # cap at last 40 turns to stay within token budget
        prompt = (
            f"Summarize this conversation in 1-2 sentences in {lang}. "
            "Focus on what the user accomplished or discussed. "
            "Output ONLY the summary text, nothing else:\n\n" + convo
        )
        try:
            from core.llm_client import get_ai_client
            client = get_ai_client()
            resp   = await asyncio.to_thread(
                client.models.generate_content,
                model="gemini-flash-latest",
                contents=prompt,
            )
            summary = (resp.text or "").strip()
            if summary:
                save_session_summary(summary, lang)
        except Exception as e:
            print(f"[Memory] ⚠️ Session summary failed: {e}")

    # ── System monitor ──────────────────────────────────────────────────────────

    async def _run_system_monitor(self) -> None:
        """Background task: voice alerts when metrics exceed thresholds."""
        while True:
            try:
                await asyncio.sleep(10)
                alert = await asyncio.to_thread(self._sys_monitor.check)
                if not alert or (not self.session and not _is_nvidia_mode()):
                    continue
                # Don't interrupt an active conversation
                with self._speaking_lock:
                    speaking = self._is_speaking
                if speaking or (time.monotonic() - self._last_user_speech) < 10:
                    continue
                await self._inject_prompt(alert)
            except (RuntimeError, asyncio.CancelledError):
                break
            except Exception as e:
                print(f"[Monitor] ⚠️ Could not send alert: {e}")

    # ── Background monitor ──────────────────────────────────────────────────────

    async def _run_background_monitor(self) -> None:
        """Check user-configured topics once per day; speak alerts when new headlines appear."""
        try:
            await asyncio.sleep(300)          # wait 5 min after startup before first check
            while True:
                if self.session or _is_nvidia_mode():
                    # Don't interrupt if user spoke recently or JARVIS is mid-sentence
                    with self._speaking_lock:
                        speaking = self._is_speaking
                    recent_speech = (time.monotonic() - self._last_user_speech) < 30
                    if not speaking and not recent_speech:
                        try:
                            alerts = await asyncio.to_thread(monitor_check_all)
                            memory = load_memory()
                            lang_e = memory.get("identity", {}).get("language", {})
                            lang   = (lang_e.get("value", "") if isinstance(lang_e, dict) else str(lang_e)).strip() or "English"
                            for alert in alerts:
                                msg = (
                                    f"{alert}\n\n"
                                    f"Inform the user about this development naturally in {lang}. "
                                    "One brief sentence only."
                                )
                                await self._inject_prompt(msg)
                                self.ui.write_log(f"SYS: Monitor alert sent.")
                                await asyncio.sleep(6)   # gap between consecutive alerts
                        except (RuntimeError, asyncio.CancelledError):
                            break
                        except Exception as e:
                            print(f"[Monitor] ⚠️ Background check error: {e}")
                await asyncio.sleep(1800)     # check every 30 minutes
        except (RuntimeError, asyncio.CancelledError):
            return

    # ── Proactive mode ──────────────────────────────────────────────────────────

    async def _run_proactive_mode(self) -> None:
        """
        Background task: periodically checks if the user has been silent long enough,
        then hands time + memory context to Gemini so it can decide what (if anything)
        to say proactively. No hardcoded rules — Gemini makes the call.
        """
        while True:
            try:
                await asyncio.sleep(60)   # evaluate once per minute

                if not self.session and not _is_nvidia_mode():
                    continue

                with self._speaking_lock:
                    speaking = self._is_speaking
                if speaking:
                    continue

                if not self._proactive.should_trigger(self._last_user_speech):
                    continue

                self._proactive.mark_triggered()

                memory       = await asyncio.to_thread(load_memory)
                monitors     = await asyncio.to_thread(list_monitors)
                recent_turns = self._session_log[-8:] if self._session_log else []
                prompt = self._proactive.build_prompt(
                    memory       = memory,
                    monitors     = monitors or None,
                    recent_turns = recent_turns or None,
                )
                await self._inject_prompt(prompt)
                self.ui.write_log("SYS: Proactive check-in.")
            except (RuntimeError, asyncio.CancelledError):
                break
            except Exception as e:
                print(f"[Proactive] ⚠️ {e}")

    # ── Phone audio relay ────────────────────────────────────────────────────────

    async def _relay_phone_audio(self) -> None:
        """Forward phone mic PCM chunks from dashboard queue into the Gemini Live session."""
        q = self._dashboard._phone_audio_queue
        while True:
            try:
                chunk = await asyncio.wait_for(q.get(), timeout=1.0)
            except asyncio.TimeoutError:
                # No audio for 1 s → phone mic inactive, give PC mic back
                self._phone_active = False
                continue
            self._phone_active = True   # phone is streaming — silence PC mic
            with self._speaking_lock:
                speaking = self._is_speaking
            if not speaking and not self.ui.muted:
                try:
                    self.out_queue.put_nowait(chunk)
                except asyncio.QueueFull:
                    pass

    def _on_phone_connected(self) -> None:
        self.ui.write_log("SYS: Phone connected via Remote Dashboard.")
        self.ui.notify_phone_connected()

    # ── dashboard command relay ─────────────────────────────────────────────

    async def _process_dashboard_commands(self) -> None:
        while True:
            try:
                text = await asyncio.wait_for(
                    self._dashboard._command_queue.get(), timeout=0.5
                )
                if not text:
                    continue
                if _is_nvidia_mode():
                    if self._nvidia_text_queue:
                        await self._nvidia_text_queue.put(text)
                        self.ui.write_log(f"[Web]: {text}")
                    continue
                # Wait up to 8s for session to become ready after a wake
                for _ in range(80):
                    if self.session:
                        break
                    await asyncio.sleep(0.1)
                if self.session:
                    await self.session.send_client_content(
                        turns={"parts": [{"text": text}]},
                        turn_complete=True,
                    )
                    self.ui.write_log(f"[Web]: {text}")
                else:
                    print(f"[Dashboard] Dropped command (no session): {text}")
            except asyncio.TimeoutError:
                pass
            except Exception as e:
                print(f"[Dashboard] Command error: {e}")
                await asyncio.sleep(0.5)

    # ── NVIDIA NIM Mode ─────────────────────────────────────────────────────

    async def _run_nvidia_mode(self):
        """
        Standalone voice assistant engine powered 100% by NVIDIA NIM API (or local Ollama).
        Uses faster-whisper on CUDA for speech-to-text, NVIDIA NIM for tool reasoning,
        NVIDIA vision NIM for camera/screen analysis, and EdgeTTS/Kokoro for speech synthesis.
        """
        from core.tts import create_tts_player
        from core.stt import WhisperSTT
        from core.llm_client import call_llm, call_llm_stream, call_llm_vision, get_llm_settings
        import numpy as np

        while True:
            try:
                print("[JARVIS] Initializing NVIDIA NIM Mode...")
                self.ui.set_state("THINKING")
                self.ui.write_log("SYS: Initializing NVIDIA NIM Mode...")

                _cfg = {}
                try:
                    with open(API_CONFIG_PATH, "r", encoding="utf-8") as f:
                        _cfg = json.load(f)
                except Exception:
                    _cfg = {}

                self._asst_name = (_cfg.get("assistant_name") or "JARVIS").strip()
                self._user_name = (_cfg.get("user_name") or "").strip()
                self._tts_player = create_tts_player(_cfg)
                whisper_model_name = _cfg.get("whisper_model", "base")
                nvidia_model = _cfg.get("llm_model", "nvidia/nemotron-3-super-120b-a12b")
                stt = await asyncio.to_thread(WhisperSTT, model_name=whisper_model_name)

                all_tools = TOOL_DECLARATIONS + self._plugin_registry.get_tool_declarations()
                openai_tools = _to_openai_tools(all_tools)

                system_instruction = self._build_system_instruction_text()
                messages: list[dict] = [
                    {"role": "system", "content": system_instruction}
                ]

                self.audio_in_queue = asyncio.Queue()
                self._nvidia_text_queue = asyncio.Queue()
                detected_audio_queue = asyncio.Queue()

                self.ui.set_state("LISTENING")
                self.ui.write_log("SYS: JARVIS online (NVIDIA NIM Mode).")
                print("[JARVIS] Online (NVIDIA NIM Mode).")

                if self._dashboard:
                    await self._dashboard.broadcast({"type": "status", "state": "active"})

                # Background monitors
                asyncio.create_task(self._run_system_monitor())
                asyncio.create_task(self._run_background_monitor())
                asyncio.create_task(self._run_proactive_mode())

                # Morning briefing if enabled
                if not self._briefing_sent and get_brief_enabled():
                    self._briefing_sent = True
                    asyncio.create_task(self._send_startup_briefing())

                # Audio listener task via robust AudioInputPipeline
                loop = asyncio.get_event_loop()
                cfg = _read_config()
                pref_mic = cfg.get("microphone_device", "auto")
                vad_sens = float(cfg.get("vad_sensitivity", 2.5))
                min_speech_dur = float(cfg.get("min_speech_duration", 0.35))
                silence_dur = float(cfg.get("silence_duration", 0.65))

                detected_audio_queue = asyncio.Queue()

                def _on_speech_completed(audio_16k: np.ndarray, duration: float):
                    loop.call_soon_threadsafe(detected_audio_queue.put_nowait, (audio_16k, duration))

                def _on_state_change(state: str):
                    if not self._is_speaking and not self.ui.muted:
                        self.ui.set_state(state)

                def _transcribe_sync(chunk: np.ndarray) -> str:
                    res = stt.transcribe(chunk, beam_size=3)
                    if isinstance(res, str):
                        return res.strip()
                    if isinstance(res, tuple):
                        segments, _ = res
                        return " ".join(s.text for s in segments).strip()
                    return str(res).strip()

                async def _transcribe_worker():
                    while True:
                        audio_np, dur = await detected_audio_queue.get()
                        print(f"[STT] STT STARTED: Segment duration: {dur:.2f}s ({len(audio_np)} samples)")
                        self.ui.set_state("TRANSCRIBING")
                        try:
                            raw_text = await asyncio.to_thread(_transcribe_sync, audio_np)
                            text = raw_text.strip()
                            print(f"[STT] STT COMPLETED")
                            if text and len(text) > 1 and not text.startswith(("[", "(")):
                                print(f"USER SPEECH: '{raw_text}'")
                                print(f"TRANSCRIPT:  '{text}'")
                                await self._nvidia_text_queue.put(text)
                            else:
                                print(f"[STT] STT RESULT EMPTY (No recognizable speech in {dur:.2f}s)")
                                if not self.ui.muted:
                                    self.ui.set_state("LISTENING")
                        except Exception as e:
                            print(f"[STT] ❌ Whisper transcription error: {e}")
                            if not self.ui.muted:
                                self.ui.set_state("LISTENING")

                transcribe_task = asyncio.create_task(_transcribe_worker())

                audio_pipeline = AudioInputPipeline(
                    preferred_device=pref_mic,
                    vad_sensitivity=vad_sens,
                    min_speech_duration=min_speech_dur,
                    silence_duration=silence_dur,
                    on_speech_completed=_on_speech_completed,
                    on_state_change=_on_state_change,
                    on_error=lambda err: self.ui.write_log(f"AUDIO ERR: {err}"),
                )
                self._audio_pipeline = audio_pipeline
                try:
                    audio_pipeline.start()
                    self.ui.write_log(f"SYS: Audio active on {audio_pipeline.device_name}")
                except Exception as e:
                    print(f"[JARVIS] ⚠️ Mic input stream failed: {e}")
                    self.ui.write_log(f"SYS: ⚠️ Mic failed: {e}")

                try:
                    while True:
                        text = await self._nvidia_text_queue.get()
                        if not text:
                            continue

                        if self._interrupted:
                            self._interrupted = False

                        self._last_user_speech = time.monotonic()
                        self.ui.write_log(f"You: {text}")
                        self._session_log.append(f"User: {text}")
                        if self._dashboard:
                            asyncio.create_task(self._dashboard.broadcast({
                                "type": "log", "speaker": "user",
                                "text": text,
                                "ts": datetime.now().isoformat(),
                            }))

                        # ── Fast-path check (sub-10ms response for local queries) ──
                        fast_reply = self._check_fast_path(text)
                        if fast_reply:
                            messages.append({"role": "user", "content": text})
                            messages.append({"role": "assistant", "content": fast_reply})
                            self.ui.write_log(f"{self._asst_name}: {fast_reply}")
                            self._session_log.append(f"{self._asst_name}: {fast_reply}")
                            if self._dashboard:
                                asyncio.create_task(self._dashboard.broadcast({
                                    "type": "log", "speaker": "jarvis",
                                    "text": fast_reply,
                                    "ts": datetime.now().isoformat(),
                                }))
                            self.set_speaking(True)
                            audio_pipeline.pause()
                            self.ui.set_state("SPEAKING")
                            print(f"[FAST-PATH] Speaking immediate reply: '{fast_reply}'")
                            try:
                                await asyncio.to_thread(self._tts_player.speak, fast_reply)
                            finally:
                                self.set_speaking(False)
                                await asyncio.sleep(0.35)
                                audio_pipeline.resume()
                            if not self.ui.muted:
                                self.ui.set_state("LISTENING")
                            continue

                        self.ui.set_state("THINKING")
                        messages.append({"role": "user", "content": text})

                        if len(messages) > 25:
                            messages = [messages[0]] + messages[-20:]

                        while True:
                            if self._interrupted:
                                self._interrupted = False
                                break

                            # Selective tool routing: omit tool schemas for conversational queries
                            needs_tools = self._needs_tools(text) or (len(messages) > 0 and messages[-1].get("role") == "tool")
                            active_tools = openai_tools if needs_tools else None

                            try:
                                _, active_model = get_llm_settings()
                                print(f"[NVIDIA] STREAM REQUEST STARTED (Model: {active_model}, Tools: {bool(active_tools)})")
                                t_llm0 = time.time()

                                stream_queue: asyncio.Queue = asyncio.Queue()
                                loop = asyncio.get_running_loop()

                                def _stream_producer():
                                    try:
                                        for item in call_llm_stream(messages, active_tools):
                                            loop.call_soon_threadsafe(stream_queue.put_nowait, item)
                                    except Exception as exc:
                                        loop.call_soon_threadsafe(stream_queue.put_nowait, {"type": "error", "error": exc})
                                    finally:
                                        loop.call_soon_threadsafe(stream_queue.put_nowait, {"type": "end"})

                                threading.Thread(target=_stream_producer, daemon=True).start()

                                content = ""
                                tool_calls = []
                                tts_queue: _queue.Queue = _queue.Queue()
                                tts_task = None
                                tts_started = False

                                while True:
                                    event = await stream_queue.get()
                                    ev_type = event.get("type")

                                    if ev_type == "error":
                                        err = event.get("error")
                                        print(f"[NVIDIA] LLM stream error: {err}")
                                        self.speak_error("NVIDIA NIM", str(err))
                                        break

                                    elif ev_type == "sentence":
                                        sent_text = event.get("text", "").strip()
                                        if sent_text:
                                            if not tts_started:
                                                tts_started = True
                                                self.set_speaking(True)
                                                audio_pipeline.pause()
                                                self.ui.set_state("SPEAKING")
                                                print(f"[TTS] STREAM FIRST SENTENCE: '{sent_text[:50]}' (TTFT: {time.time()-t_llm0:.2f}s)")
                                                tts_task = asyncio.to_thread(self._tts_player.play_stream, tts_queue)
                                            tts_queue.put(sent_text)

                                    elif ev_type == "done":
                                        content = event.get("content", "").strip()
                                        tool_calls = event.get("tool_calls", [])
                                        tts_queue.put(None)
                                        print(f"[NVIDIA] RESPONSE RECEIVED in {time.time()-t_llm0:.2f}s")

                                    elif ev_type == "end":
                                        break

                            except Exception as e:
                                print(f"[NVIDIA] LLM call error: {e}")
                                self.speak_error("NVIDIA NIM", str(e))
                                break

                            if tool_calls:
                                if tts_task:
                                    self._tts_player.stop()
                                    try:
                                        await tts_task
                                    except Exception:
                                        pass
                                    self.set_speaking(False)

                                raw_tc_list = []
                                for tc in tool_calls:
                                    fn = tc["function"]
                                    raw_tc_list.append({
                                        "id": tc["id"],
                                        "type": "function",
                                        "function": {
                                            "name": fn["name"],
                                            "arguments": json.dumps(fn["arguments"]) if isinstance(fn["arguments"], dict) else str(fn["arguments"]),
                                        },
                                    })
                                messages.append({
                                    "role": "assistant",
                                    "content": content or None,
                                    "tool_calls": raw_tc_list,
                                })

                                for tc in tool_calls:
                                    fn_name = tc["function"]["name"]
                                    fn_args = tc["function"]["arguments"]
                                    if isinstance(fn_args, str):
                                        try:
                                            fn_args = json.loads(fn_args)
                                        except Exception:
                                            fn_args = {}
                                    tc_id = tc["id"]
                                    self.ui.write_log(f"SYS: Calling tool {fn_name}...")
                                    wrapped_fc = ToolCallWrapper(fn_name, fn_args, tc_id)
                                    tool_resp = await self._execute_tool(wrapped_fc)

                                    res_str = ""
                                    if hasattr(tool_resp, "response") and isinstance(tool_resp.response, dict):
                                        res_str = str(tool_resp.response.get("result", ""))
                                    else:
                                        res_str = str(tool_resp)

                                    if self._pending_vision:
                                        img_b, mime_t, v_text, angle = self._pending_vision
                                        self._pending_vision = None
                                        self.ui.write_log(f"[Vision]: Processing {angle} with NVIDIA Vision NIM...")
                                        try:
                                            v_result = await asyncio.to_thread(
                                                call_llm_vision,
                                                img_b,
                                                f"User question about this {angle}: {v_text}",
                                                mime_t,
                                            )
                                            res_str = f"[Vision Analysis of {angle}]: {v_result}"
                                        except Exception as ve:
                                            res_str = f"[Vision Error on {angle}]: {ve}"
                                        if self._vision_cam_active:
                                            self.ui.stop_camera_stream()
                                            self._vision_cam_active = False
                                        self._vision_busy = False

                                    messages.append({
                                        "role": "tool",
                                        "tool_call_id": tc_id,
                                        "name": fn_name,
                                        "content": res_str,
                                    })

                                continue

                            else:
                                if content:
                                    messages.append({"role": "assistant", "content": content})
                                    self.ui.write_log(f"{self._asst_name}: {content}")
                                    self._session_log.append(f"{self._asst_name}: {content}")
                                    if self._dashboard:
                                        asyncio.create_task(self._dashboard.broadcast({
                                            "type": "log", "speaker": "jarvis",
                                            "text": content,
                                            "ts": datetime.now().isoformat(),
                                        }))

                                    if tts_task:
                                        try:
                                            await tts_task
                                        finally:
                                            self.set_speaking(False)
                                            await asyncio.sleep(0.35)
                                            audio_pipeline.resume()
                                    else:
                                        self.set_speaking(True)
                                        audio_pipeline.pause()
                                        self.ui.set_state("SPEAKING")
                                        print(f"[TTS] STARTED: '{content[:60]}...'")
                                        try:
                                            await asyncio.to_thread(self._tts_player.speak, content)
                                        finally:
                                            self.set_speaking(False)
                                            await asyncio.sleep(0.35)
                                            audio_pipeline.resume()
                                break

                        if not self.ui.muted:
                            self.ui.set_state("LISTENING")

                finally:
                    transcribe_task.cancel()
                    audio_pipeline.stop()
                    self._audio_pipeline = None
            except Exception as e:
                print(f"[JARVIS-NVIDIA] Unexpected error: {e}")
                traceback.print_exc()
                self.ui.write_log(f"ERR: {e}")
                await asyncio.sleep(3)

    # ── main loop ───────────────────────────────────────────────────────────

    async def run(self):
        self._loop = asyncio.get_event_loop()

        # Start dashboard (optional — needs: pip install fastapi "uvicorn[standard]" cryptography)
        try:
            from dashboard.server import DashboardServer
            self._dashboard = DashboardServer()
            self._dashboard.set_connect_callback(self._on_phone_connected)
            asyncio.create_task(self._dashboard.serve())
            # Runs for the whole lifetime, not just inside an active session
            asyncio.create_task(self._process_dashboard_commands())
        except Exception as e:
            print(f"[Dashboard] Disabled: {e}")
            self._dashboard = None

        if _is_nvidia_mode():
            await self._run_nvidia_mode()
            return

        while True:
            try:
                print("[JARVIS] Connecting...")
                self.ui.set_state("THINKING")
                config = self._build_config()

                # Fresh client on every reconnect — avoids stale HTTP session state
                # v1alpha carries the enhanced audio features (affective dialog,
                # proactive audio); if they get rejected we fall back to v1beta.
                client = genai.Client(
                    api_key=_get_api_key(),
                    http_options={"api_version": "v1alpha" if self._enhanced_live else "v1beta"}
                )

                async with (
                    client.aio.live.connect(model=LIVE_MODEL, config=config) as session,
                    asyncio.TaskGroup() as tg,
                ):
                    self.session          = session
                    self.audio_in_queue   = asyncio.Queue()
                    self.out_queue        = asyncio.Queue(maxsize=200)
                    self._turn_done_event = asyncio.Event()

                    # Reset transient state that must not carry over from a previous session
                    self._pending_vision       = None
                    self._vision_cam_active    = False
                    self._vision_close_pending = False
                    self._vision_busy          = False
                    self._vision_last_time     = 0.0
                    self._interrupted          = False

                    print("[JARVIS] Connected.")
                    self.ui.set_state("LISTENING")
                    self.ui.write_log("SYS: JARVIS online.")

                    if self._dashboard:
                        await self._dashboard.broadcast({"type": "status", "state": "active"})

                    tg.create_task(self._send_realtime())
                    tg.create_task(self._listen_audio())
                    tg.create_task(self._receive_audio())
                    tg.create_task(self._play_audio())
                    tg.create_task(self._run_system_monitor())
                    tg.create_task(self._run_background_monitor())
                    tg.create_task(self._run_proactive_mode())
                    if self._dashboard:
                        tg.create_task(self._relay_phone_audio())

                    # Morning briefing — fires once per process launch (if enabled)
                    if not self._briefing_sent and get_brief_enabled():
                        self._briefing_sent = True
                        tg.create_task(self._send_startup_briefing())

            except KeyboardInterrupt:
                raise
            except SystemExit:
                raise
            except BaseException as e:
                # Catches both Exception and BaseExceptionGroup (Python 3.11+
                # TaskGroup raises BaseExceptionGroup when tasks are cancelled
                # externally, which `except Exception` would miss, letting the
                # exception escape the while-loop and causing asyncio.run() to
                # start shutdown — resulting in "executor after shutdown" errors).
                err_str = str(e)
                print(f"[JARVIS] Error ({type(e).__name__}): {e}")
                traceback.print_exc()

                # Enhanced audio features rejected by the server (preview API
                # drift) — drop them and reconnect with the plain config.
                if self._enhanced_live and (
                    "INVALID_ARGUMENT" in err_str
                    or "affective" in err_str.lower()
                    or "proactiv" in err_str.lower()
                    or "Unknown name" in err_str
                    or "unexpected keyword" in err_str
                ):
                    self._enhanced_live = False
                    self.ui.write_log(
                        "SYS: Advanced audio features unavailable — reconnecting without them."
                    )
                    continue

                # Invalid API key — stop hammering the API, prompt re-configuration
                if "API key not valid" in err_str or "1007" in err_str:
                    self.ui.write_log("ERR: API key invalid — please re-enter your key.")
                    self.ui.set_state("SLEEPING")
                    self.ui.prompt_reconfig()
                    while not self.ui._win._ready:
                        await asyncio.sleep(1)
                    print("[JARVIS] New API key saved — reconnecting...")
                    _conn_backoff = 3
                    continue

                # Network / timeout errors — log clearly and back off
                is_net_err = any(k in err_str for k in (
                    "TimeoutError", "timed out", "getaddrinfo", "CancelledError",
                    "ConnectionRefusedError", "OSError", "Cannot connect",
                ))
                if is_net_err:
                    _conn_backoff = min(getattr(self, "_conn_backoff", 3) * 2, 60)
                    self._conn_backoff = _conn_backoff
                    self.ui.write_log(
                        f"NET: Bağlantı kurulamadı — {_conn_backoff}s sonra tekrar deneniyor. "
                        "(VPN gerekiyor olabilir)"
                    )
                else:
                    self._conn_backoff = 3
            finally:
                self.session = None
                # Only save if there was a real conversation (≥3 turns)
                if len(self._session_log) >= 3:
                    asyncio.create_task(self._save_session_summary())

            self.set_speaking(False)
            self.ui.set_state("SLEEPING")

            if self._dashboard:
                await self._dashboard.broadcast({"type": "status", "state": "sleeping"})

            delay = getattr(self, "_conn_backoff", 3)
            print(f"[JARVIS] Reconnecting in {delay}s...")
            await asyncio.sleep(delay)

def main():
    if "--help" in sys.argv or "-h" in sys.argv:
        print("\nJARVIS Voice & Desktop AI Assistant")
        print("Usage:")
        print("  python main.py               Start JARVIS desktop interface")
        print("  python main.py --test-mic    Run standalone microphone & STT diagnostic")
        print("  python main.py --help        Show this help message\n")
        return

    if "--test-mic" in sys.argv:
        dur = 5.0
        for arg in sys.argv[1:]:
            if arg != "--test-mic" and arg.replace(".", "", 1).isdigit():
                dur = float(arg)
                break
        run_standalone_mic_test(duration=dur)
        return

    ui = JarvisUI("face.png")

    def runner():
        ui.wait_for_api_key()
        jarvis = JarvisLive(ui)
        try:
            asyncio.run(jarvis.run())
        except KeyboardInterrupt:
            print("\n🔴 Shutting down...")

    threading.Thread(target=runner, daemon=True).start()
    ui.root.mainloop()

if __name__ == "__main__":
    main()