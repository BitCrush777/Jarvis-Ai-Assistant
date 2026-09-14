import threading
import time
import asyncio
import sys
import json
from pathlib import Path

# Add core actions to path so we can reuse Mark-LI's built-in send_message
sys.path.append(str(Path(__file__).resolve().parent.parent))
try:
    from actions.send_message import _send_whatsapp
except ImportError:
    _send_whatsapp = None

# Try importing winsdk for native Windows Notification reading
try:
    from winsdk.windows.ui.notifications.management import UserNotificationListener
    from winsdk.windows.ui.notifications import NotificationKinds
    _WINSDK_AVAILABLE = True
except ImportError:
    _WINSDK_AVAILABLE = False


PLUGIN = {
    "name": "whatsapp_monitor",
    "description": (
        "Turn on or off the WhatsApp auto-responder. "
        "When active, Jarvis will read Windows System Notifications for WhatsApp messages, "
        "generate a smart AI response based on the message content, and automatically send it."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "status": {
                "type": "STRING",
                "description": "'on' to start monitoring, 'off' to stop monitoring."
            }
        },
        "required": ["status"],
    },
}

_monitor_thread = None
_monitoring_active = False
_replied_notification_ids = set()

def _generate_dynamic_reply(sender, message):
    try:
        from core.llm_client import call_llm_text

        prompt = (
            f"You are Jarvis, Deepak's personal AI assistant. Deepak is currently busy working. "
            f"You received a WhatsApp message from '{sender}' which says: '{message}'. "
            f"Write a short, natural, and polite reply on Deepak's behalf. "
            f"Acknowledge what they said. If it is an important update (like a meeting, emergency, or request), "
            f"tell them you will convey it to Deepak immediately. "
            f"Keep it under 2 sentences. Reply directly as Jarvis."
        )

        reply = call_llm_text(prompt)
        return reply.strip()
    except Exception as e:
        print(f"[WhatsApp Monitor] AI generation failed: {e}")
        return f"Hello, I am Jarvis. Deepak is currently busy, but I will convey your message to him."

async def _get_whatsapp_notifications():
    if not _WINSDK_AVAILABLE:
        return []
        
    listener = UserNotificationListener.current
    access = await listener.request_access_async()
    
    if access != 1:  # 1 means Allowed
        print("[WhatsApp Monitor] Windows Notification access denied by user/system.")
        return []
        
    notifs = await listener.get_notifications_async(NotificationKinds.TOAST)
    
    results = []
    for n in notifs:
        try:
            app_name = n.app_info.display_info.display_name
            if "WhatsApp" in app_name:
                bindings = n.notification.visual.bindings
                texts = []
                for b in bindings:
                    for t in b.get_text_elements():
                        texts.append(t.text)
                
                if texts:
                    results.append({
                        "id": n.id,
                        "sender": texts[0],
                        "message": texts[1] if len(texts) > 1 else "",
                    })
        except Exception as e:
            pass
            
    return results

def _whatsapp_monitor_loop(player):
    global _monitoring_active, _replied_notification_ids
    
    print("[WhatsApp Monitor] Background thread started with Dynamic AI Replies.")
    
    if player:
        try:
            player.write_log("JARVIS: WhatsApp auto-reply with Smart AI is now active.")
        except Exception:
            pass

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    while _monitoring_active:
        if _WINSDK_AVAILABLE and _send_whatsapp:
            try:
                notifications = loop.run_until_complete(_get_whatsapp_notifications())
                
                for notif in notifications:
                    notif_id = notif["id"]
                    sender = notif["sender"]
                    message = notif["message"]
                    
                    if notif_id not in _replied_notification_ids:
                        _replied_notification_ids.add(notif_id)
                        
                        print(f"[WhatsApp Monitor] New message from {sender}: {message}")
                        
                        # 1. Generate Smart Reply
                        reply_text = _generate_dynamic_reply(sender, message)
                        
                        # 2. Open WhatsApp and send
                        result_status = _send_whatsapp(sender, reply_text)
                        
                        # 3. Unfocus the chat so follow-up messages trigger notifications again!
                        try:
                            import pyautogui
                            time.sleep(1)
                            pyautogui.press('esc')  # Deselect the chat
                            time.sleep(0.5)
                            pyautogui.hotkey('win', 'down') # Minimize the window
                        except Exception as e:
                            print(f"[WhatsApp Monitor] Could not minimize window: {e}")
                        
                        # 4. Notify Deepak via JARVIS UI
                        alert_msg = f"Sir, '{sender}' sent you a message: '{message}'. I replied with: '{reply_text}'"
                        if player:
                            player.write_log(f"JARVIS: 📨 {alert_msg}")
                            # Writing twice to make sure it's prominent in the UI logs
                            print(f"[JARVIS UI ALERT] {alert_msg}")
                            
                            # 5. INJECT into Jarvis's Brain (LLM Context) so he remembers it!
                            if hasattr(player, 'on_text_command') and callable(player.on_text_command):
                                internal_memory = (
                                    f"[SYSTEM_ALERT] You just automatically replied to a WhatsApp message in the background.\n"
                                    f"Sender: {sender}\n"
                                    f"Their message: {message}\n"
                                    f"Your reply: {reply_text}\n\n"
                                    f"Keep this in your memory. DO NOT speak or reply to this alert out loud unless Deepak explicitly asks if you sent any messages."
                                )
                                try:
                                    player.on_text_command(internal_memory)
                                except Exception as e:
                                    print(f"[WhatsApp Monitor] Failed to update Jarvis memory: {e}")
                            
            except Exception as e:
                print(f"[WhatsApp Monitor] Error processing notifications: {e}")
        else:
            time.sleep(5)
            
        time.sleep(3)

    loop.close()
    print("[WhatsApp Monitor] Background thread stopped.")

def run(parameters: dict, player=None, session_memory=None) -> str:
    global _monitor_thread, _monitoring_active
    
    status = parameters.get("status", "off").lower()
    
    if status == "on":
        if _monitoring_active:
            return "WhatsApp smart auto-reply is already monitoring."
            
        _monitoring_active = True
        _monitor_thread = threading.Thread(
            target=_whatsapp_monitor_loop,
            args=(player,),
            daemon=True
        )
        _monitor_thread.start()
        
        missing_libs = []
        if not _WINSDK_AVAILABLE:
            missing_libs.append("winsdk")
        if not _GENAI_AVAILABLE:
            missing_libs.append("google-generativeai")
            
        if missing_libs:
            return f"Monitoring started, but missing libraries: {', '.join(missing_libs)}. Run pip install."
            
        return "WhatsApp smart auto-reply enabled. I will read incoming messages, think of a contextual reply, and send it on your behalf."
        
    elif status == "off":
        if not _monitoring_active:
            return "WhatsApp monitoring is already off."
            
        _monitoring_active = False
        return "I have stopped monitoring WhatsApp."
        
    return "Invalid status. Use 'on' or 'off'."
