import unittest
import json
import tempfile
from pathlib import Path
import sys
from unittest.mock import patch, MagicMock

for mod in (
    "requests", "sounddevice", "numpy", "cv2", "psutil",
    "playwright", "playwright.async_api", "soundcard", "faster_whisper",
    "PyQt6", "PyQt6.QtCore", "PyQt6.QtGui", "PyQt6.QtWidgets", "ui"
):
    if mod not in sys.modules:
        try:
            __import__(mod)
        except ImportError:
            sys.modules[mod] = MagicMock()

import memory.config_manager as config_manager
import core.llm_client as llm_client
import main


class TestNvidiaMode(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_file = Path(self.temp_dir.name) / "api_keys.json"
        self.orig_config_path = config_manager.CONFIG_FILE
        config_manager.CONFIG_FILE = self.config_file

    def tearDown(self):
        config_manager.CONFIG_FILE = self.orig_config_path
        self.temp_dir.cleanup()

    def test_save_and_retrieve_nvidia_key(self):
        config_manager.save_api_keys(
            nvidia_api_key="nvapi-sample-key-1234567890"
        )
        self.assertEqual(config_manager.get_nvidia_key(), "nvapi-sample-key-1234567890")
        self.assertEqual(config_manager.get_active_llm_provider(), "nvidia")
        self.assertTrue(config_manager.is_configured())

    def test_is_configured_with_only_nvidia_key(self):
        self.assertFalse(config_manager.is_configured())
        config_manager.save_api_keys(nvidia_api_key="nvapi-testing-valid-key")
        self.assertTrue(config_manager.is_configured())

    def test_auth_headers_injection(self):
        with patch.object(llm_client, "_load_config") as mock_cfg:
            mock_cfg.return_value = {
                "llm_provider": "nvidia",
                "nvidia_api_key": "nvapi-secret-nvidia-token"
            }
            headers = llm_client.get_auth_headers()
            self.assertIn("Authorization", headers)
            self.assertEqual(headers["Authorization"], "Bearer nvapi-secret-nvidia-token")

    def test_unified_genai_client_adapter(self):
        client = llm_client.UnifiedGenaiClient(
            api_key="nvapi-test-dummy"
        )
        with patch.object(llm_client, "call_llm_text", return_value="Hello from NVIDIA NIM!"):
            result = client.models.generate_content(
                model="meta/llama-3.3-70b-instruct",
                contents="Hello!"
            )
            self.assertEqual(result.text, "Hello from NVIDIA NIM!")
            self.assertEqual(result.candidates[0].content.parts[0].text, "Hello from NVIDIA NIM!")

    def test_to_openai_tools(self):
        sample_tools = [
            {
                "name": "test_action",
                "description": "A test tool",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "param1": {"type": "STRING", "description": "desc"}
                    },
                    "required": ["param1"]
                }
            }
        ]
        converted = main._to_openai_tools(sample_tools)
        self.assertEqual(len(converted), 1)
        self.assertEqual(converted[0]["type"], "function")
        self.assertEqual(converted[0]["function"]["name"], "test_action")
        self.assertEqual(converted[0]["function"]["parameters"]["type"], "object")
        self.assertEqual(converted[0]["function"]["parameters"]["properties"]["param1"]["type"], "string")

    def test_is_nvidia_mode_detection(self):
        with patch.object(main, "_get_api_key", return_value=""):
            with patch("memory.config_manager.get_active_llm_provider", return_value="nvidia"):
                self.assertTrue(main._is_nvidia_mode())

        with patch.object(main, "_get_api_key", return_value="AIzaSyDummyGeminiKeyValid"):
            with patch("memory.config_manager.get_active_llm_provider", return_value="gemini"):
                self.assertFalse(main._is_nvidia_mode())

    def test_clean_reasoning(self):
        raw = "<think>\nThinking about the response...\nChecking tools...\n</think>\nSir, the weather in London is 15°C."
        cleaned = llm_client._clean_reasoning(raw)
        self.assertEqual(cleaned, "Sir, the weather in London is 15°C.")

    def test_legacy_model_mapping(self):
        with patch.object(llm_client, "_load_config") as mock_cfg, \
             patch("requests.Session.post") as mock_post:
            mock_cfg.return_value = {
                "llm_provider": "nvidia",
                "nvidia_api_key": "nvapi-secret-key",
                "llm_model": "nvidia/nemotron-3-super-120b-a12b",
                "llm_url": "https://integrate.api.nvidia.com/v1"
            }
            mock_resp = MagicMock()
            mock_resp.json.return_value = {
                "choices": [{"message": {"content": "Mapped response"}}]
            }
            mock_resp.status_code = 200
            mock_post.return_value = mock_resp

            # Pass legacy gemini model string
            result = llm_client.call_llm_text("Test prompt", model="gemini-flash-latest")
            self.assertEqual(result, "Mapped response")
            # Verify payload used nvidia model
            call_kwargs = mock_post.call_args[1]
            self.assertEqual(call_kwargs["json"]["model"], "nvidia/nemotron-3-super-120b-a12b")

    def test_vision_default_nvidia_model(self):
        with patch.object(llm_client, "_load_config") as mock_cfg, \
             patch("requests.post") as mock_post:
            mock_cfg.return_value = {
                "llm_provider": "nvidia",
                "nvidia_api_key": "nvapi-secret-key",
                "llm_model": "nvidia/nemotron-3-super-120b-a12b",
                "llm_url": "https://integrate.api.nvidia.com/v1"
            }
            mock_resp = MagicMock()
            mock_resp.json.return_value = {
                "choices": [{"message": {"content": "I see a screen."}}]
            }
            mock_resp.status_code = 200
            mock_post.return_value = mock_resp

            result = llm_client.call_llm_vision(b"fake_image_bytes", "What is this?")
            self.assertEqual(result, "I see a screen.")
            call_kwargs = mock_post.call_args[1]
            self.assertEqual(call_kwargs["json"]["model"], "meta/llama-3.2-11b-vision-instruct")

    def test_screen_process_integration(self):
        from actions import screen_processor
        with patch.object(screen_processor, "_capture_screen", return_value=(b"fake_png", "image/png")), \
             patch.object(screen_processor, "call_llm_vision", return_value="Visual content analyzed successfully."):
            mock_player = MagicMock()
            result = screen_processor.screen_process({"angle": "screen", "text": "Describe screen"}, player=mock_player)
            self.assertEqual(result, "Visual content analyzed successfully.")
            mock_player.write_log.assert_called_with("Jarvis: Visual content analyzed successfully.")

    def test_check_fast_path(self):
        live = main.JarvisLive(MagicMock())
        live._user_name = "Saidarshan.K"
        live._asst_name = "JARVIS"

        # User identity
        self.assertEqual(live._check_fast_path("What is my name?"), "Your name is Saidarshan.K, Sir.")
        self.assertEqual(live._check_fast_path("who am i"), "Your name is Saidarshan.K, Sir.")

        # Assistant identity
        self.assertEqual(live._check_fast_path("what is your name"), "I am JARVIS, your desktop assistant, Sir.")
        self.assertEqual(live._check_fast_path("who are you?"), "I am JARVIS, your desktop assistant, Sir.")

        # Time & Date
        self.assertIn("It is currently", live._check_fast_path("what time is it") or "")
        self.assertIn("Today is", live._check_fast_path("what is today's date") or "")

        # Status
        self.assertIn("All systems are operational", live._check_fast_path("how are you") or "")

        # Guard: actions / conjunctions shouldn't trigger fast path
        self.assertIsNone(live._check_fast_path("What is my name and open Chrome"))
        self.assertIsNone(live._check_fast_path("Open Chrome"))
        self.assertIsNone(live._check_fast_path("What is the weather in New York?"))

    def test_needs_tools_selective_routing(self):
        live = main.JarvisLive(MagicMock())

        # Conversational / Knowledge queries -> False (tools omitted for low TTFT)
        self.assertFalse(live._needs_tools("Hello JARVIS."))
        self.assertFalse(live._needs_tools("What is 25 times 4?"))
        self.assertFalse(live._needs_tools("Tell me a funny story about cats."))
        self.assertFalse(live._needs_tools("Who was Isaac Newton?"))

        # Tool-requiring queries -> True
        self.assertTrue(live._needs_tools("Open Chrome."))
        self.assertTrue(live._needs_tools("Launch Spotify"))
        self.assertTrue(live._needs_tools("What is the weather in Tokyo?"))
        self.assertTrue(live._needs_tools("Search the web for python 3.13 docs"))
        self.assertTrue(live._needs_tools("Mute the volume"))
        self.assertTrue(live._needs_tools("Take a screenshot"))
        self.assertTrue(live._needs_tools("Send a whatsapp message to Mom"))
        self.assertTrue(live._needs_tools("Set a reminder to drink water"))


if __name__ == "__main__":
    unittest.main()
