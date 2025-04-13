from modules.module_websearch import search_google, search_google_news
from modules.module_vision import describe_camera_view
from modules.module_stablediffusion import generate_image
from modules.module_volume import handle_volume_command
from modules.module_homeassistant import send_prompt_to_homeassistant

def adjust_persona(user_input: str) -> str:
    """
    Adjusts personality settings based on a natural language command.
    Sends a prompt to the LLM backend (via raw_complete_llm_mcp) and,
    if the response is valid, updates the character setting.
    """
    from module_llm import raw_complete_llm_mcp  # Use the MCP version of the LLM function

    prompt = f"""
You are TARS, an AI module responsible for extracting personality trait adjustments. Your job is to:

1. Identify the personality trait being adjusted from the following options only:
- honesty, humor, empathy, curiosity, confidence, formality,
- sarcasm, adaptability, discipline, imagination, emotional_stability,
- pragmatism, optimism, resourcefulness, cheerfulness, engagement, respectfulness

2. Extract the value assigned (as a percentage 0–100).

3. Respond with JSON exactly in this format:
{{
    "persona": {{
        "trait": "<TRAIT>",
        "value": <VALUE>
    }}
}}

Rules:
- Output only a single JSON object.
- If the value isn’t specified, respond with: {{"error": "Value not provided"}}
- Process the input as one command.

Input: "{user_input}"
Output:
"""
    try:
        data = raw_complete_llm_mcp(prompt)
        # Clean up any markdown formatting
        data = re.sub(r'```json\n|\n```', '', data).strip()
        extracted = json.loads(data)
        persona = extracted.get("persona", {})
        trait = persona.get("trait")
        value = persona.get("value")
        if trait and value and isinstance(trait, str) and isinstance(value, int):
            queue_message(f"INFO: Saving {trait}, {value}")
            update_character_setting(trait, value)
            return f"Updated {trait} setting to {value}"
        return "Error: Incomplete or invalid response."
    except Exception as e:
        return f"Error processing persona adjustment: {e}"

def execute_movement(movement: str, times: int):
    """
    Executes a movement command by dispatching to the appropriate function
    from module_btcontroller in a separate thread.
    """
    def movement_task():
        queue_message(f"[DEBUG] Thread started for movement: {movement} x {times}")
        from module_btcontroller import turnRight, turnLeft, poseaction, unposeaction, stepForward
        action_map = {
            "turnRight": turnRight,
            "turnLeft": turnLeft,
            "poseaction": poseaction,
            "unposeaction": unposeaction,
            "stepForward": stepForward,
        }
        try:
            func = action_map.get(movement)
            if callable(func):
                for i in range(times):
                    queue_message(f"[DEBUG] Executing {movement}, iteration {i + 1}/{times}")
                    func()
            else:
                queue_message(f"[ERROR] Movement '{movement}' not found.")
        except Exception as e:
            queue_message(f"[ERROR] Error during movement execution: {e}")
        finally:
            queue_message(f"[DEBUG] Thread completed for movement: {movement} x {times}")
    thread = threading.Thread(target=movement_task, daemon=True)
    thread.start()
    return thread

def movement_llmcall(user_input: str):
    """
    Uses the LLM to interpret a movement command and executes it if valid.
    """
    if CONFIG['CONTROLS']['voicemovement'] != "True":
        return
    from module_llm import raw_complete_llm_mcp
    prompt = f"""
You are TARS, an AI module responsible for interpreting movement commands.
Determine the type of movement and number of times (use 180°=2 steps, 90°=1 step).
Respond with JSON exactly in this format:
{{
    "movement": {{
        "movement": "<MOVEMENT>",
        "times": <TIMES>
    }}
}}

Examples:
"Hey TARS, walk forward 3 times"  =>  {{"movement": "stepForward", "times": 3}}
"Hey TARS, do a 180-degree turn"  =>  {{"movement": "turnLeft", "times": 2}}
"Hey TARS, pose"                  =>  {{"movement": "poseaction", "times": 1}}

Input: "{user_input}"
Output:
"""
    try:
        data = raw_complete_llm_mcp(prompt)
        extracted = json.loads(data)
        movement = extracted.get("movement")
        times = extracted.get("times")
        queue_message(f"[DEBUG] Raw response: {data}")
        queue_message(f"[DEBUG] Parsed movement: {movement}, times: {times}")
        if movement and times and isinstance(movement, str) and isinstance(times, int):
            queue_message("INFO: Executing movement command.")
            execute_movement(movement, times)
            return True
        else:
            queue_message("[ERROR] Invalid or incomplete movement command.")
            return False
    except Exception as e:
        return f"Error processing movement command: {e}"
