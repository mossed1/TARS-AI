import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

import requests
import base64
from PIL import Image
import tempfile
import time
import pygame
import threading
from io import BytesIO
import asyncio

# Import your configuration and message queue functions.
from modules.module_config import load_config
from modules.module_messageQue import queue_message

# Load configuration.
config = load_config()

def generate_image(prompt: str) -> str:
    """
    Generate an image based on the provided prompt using the configured image generation service. 
    Returns a status message.
    """
    result = "Image Tool not enabled"
    if config['STABLE_DIFFUSION']['enabled'] == "True":
        if config['STABLE_DIFFUSION']['service'] == "openai":
            result = get_image_from_dalle_v3(prompt)
        elif config['STABLE_DIFFUSION']['service'] == "automatic1111":
            result = get_image_from_automatic1111(prompt)
    return result

def get_image_from_dalle_v3(prompt: str) -> str:
    # Initialize the OpenAI client
    from openai import OpenAI
    client = OpenAI(api_key=config['LLM']['api_key'])
    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        # Extract the image URL
        image_url = response.data[0].url
        # Fetch the image data from the URL
        image_response = requests.get(image_url)
        image_response.raise_for_status()
        # Decode the image data into a PIL image
        image = Image.open(BytesIO(image_response.content))
        # Save the image to a temporary file after resizing to 512x512
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_png_file:
            resized_image = image.resize((512, 512))
            resized_image.save(temp_png_file, format='PNG')
            temp_png_file_path = temp_png_file.name
        # Display the image in fullscreen using a thread
        display_thread = threading.Thread(target=display_image_fullscreen, args=(temp_png_file_path,))
        display_thread.start()
        return "Image generated and displayed in fullscreen."
    except Exception as e:
        queue_message(f"Error: {e}")
        return f"Error: {e}"

def get_image_from_automatic1111(prompt: str) -> str:
    payload = {
        "prompt": prompt,
        "negative_prompt": config['STABLE_DIFFUSION']['negative_prompt'],
        "seed": int(config['STABLE_DIFFUSION']['seed']),
        "sampler_name": config['STABLE_DIFFUSION']['sampler_name'],
        "denoising_strength": float(config['STABLE_DIFFUSION']['denoising_strength']),
        "steps": int(config['STABLE_DIFFUSION']['steps']),
        "cfg_scale": float(config['STABLE_DIFFUSION']['cfg_scale']),
        "width": int(config['STABLE_DIFFUSION']['width']),
        "height": int(config['STABLE_DIFFUSION']['height']),
        "restore_faces": config.get('STABLE_DIFFUSION', 'restore_faces') == 'True',
        "override_settings_restore_afterwards": True,
    }
    url = f'{config["STABLE_DIFFUSION"]["url"]}/sdapi/v1/txt2img'
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        image_data_base64 = response.json()['images'][0]
        image_data = base64.b64decode(image_data_base64)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_png_file:
            temp_png_file.write(image_data)
            temp_png_file_path = temp_png_file.name
        display_thread = threading.Thread(target=display_image_fullscreen, args=(temp_png_file_path,))
        display_thread.start()
        return "The image has been created and displayed on screen."
    except requests.exceptions.HTTPError as err:
        queue_message(f"HTTP error occurred: {err}")
    except requests.exceptions.RequestException as e:
        queue_message(f"Error: {e}")
    return "Image generation failed."

def display_image_fullscreen(image_path: str):
    """Display an image in fullscreen (scaled to fit) for 8 seconds."""
    pygame.init()
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    time.sleep(0.1)
    screen_width, screen_height = screen.get_size()
    pygame_img = pygame.image.load(image_path)
    img_width, img_height = pygame_img.get_width(), pygame_img.get_height()
    scale_factor = min(screen_width / img_width, screen_height / img_height)
    new_width = int(img_width * scale_factor)
    new_height = int(img_height * scale_factor)
    scaled_img = pygame.transform.smoothscale(pygame_img, (new_width, new_height))
    x_pos = (screen_width - new_width) // 2
    y_pos = (screen_height - new_height) // 2
    screen.fill((0, 0, 0))
    screen.blit(scaled_img, (x_pos, y_pos))
    pygame.display.update()
    start_ticks = pygame.time.get_ticks()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        if pygame.time.get_ticks() - start_ticks > 8000:
            running = False
        pygame.display.update()
    pygame.quit()

# --- MCP Server Setup ---

from mcp.server.fastmcp import FastMCP
from mcp.server.models import InitializationOptions

# Create an MCP server instance.
mcp = FastMCP("Image Generator MCP Server")

# Expose the image generation functionality as an MCP tool.
@mcp.tool()
def generate(prompt: str) -> str:
    """
    Generate an image based on the provided prompt.
    
    This MCP tool calls the configured image generation service.
    """
    return generate_image(prompt)

# Async main entrypoint to run the MCP server.
async def main():
    await mcp.run(
        InitializationOptions(
            server_name="image_generator",
            server_version="1.0.0",
            capabilities={
                "tools": {"listChanged": True},
            },
        )
    )

if __name__ == "__main__":
    asyncio.run(main())
