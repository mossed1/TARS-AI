#!/usr/bin/env python3
import json
import subprocess
import os
import sys
import shutil

def check_install_node_tools():
    """
    Ensure that both npm and npx are installed.
    If npm is missing (Linux only), attempt to install it.
    Then, check for npx and install it if necessary.
    """
    # Check if npm is installed.
    if shutil.which("npm") is None:
        print("npm is not installed. Attempting to install npm using apt-get (Linux)...")
        if sys.platform.startswith("linux"):
            try:
                subprocess.run(["sudo", "apt-get", "update"], check=True)
                subprocess.run(["sudo", "apt-get", "install", "-y", "npm"], check=True)
            except subprocess.CalledProcessError:
                print("Failed to install npm. Please install npm manually.")
                sys.exit(1)
        else:
            print("Automatic installation of npm is not supported on this platform. Please install npm manually.")
            sys.exit(1)
        if shutil.which("npm") is None:
            print("npm still not found after installation. Exiting.")
            sys.exit(1)
        else:
            print("npm installed successfully.")
    else:
        print("npm is already installed.")

    # Check if npx is installed. npx should be bundled with npm v5.2.0+.
    if shutil.which("npx") is None:
        print("npx is not installed. Attempting to install npx using npm...")
        try:
            subprocess.run(["npm", "install", "-g", "npx"], check=True)
            if shutil.which("npx") is None:
                print("npx still not found after installation. Exiting.")
                sys.exit(1)
            else:
                print("npx installed successfully.")
        except subprocess.CalledProcessError:
            print("Failed to install npx. Please install it manually.")
            sys.exit(1)
    else:
        print("npx is already installed.")

def load_config(config_path: str) -> dict:
    """Load the JSON configuration file and return it as a dictionary."""
    with open(config_path, "r") as file:
        return json.load(file)

def launch_mcp_server(config: dict, server_name: str):
    """
    Launch the MCP server defined under 'mcpServers' for the given server_name.
    It constructs the command from the config and starts the process.
    """
    server_config = config.get("mcpServers", {}).get(server_name)
    if not server_config:
        raise ValueError(f"No configuration found for server: {server_name}")

    # Construct the command and update environment variables if provided.
    command = [server_config.get("command")] + server_config.get("args", [])
    env = os.environ.copy()
    if "env" in server_config:
        env.update(server_config["env"])

    print(f"Launching {server_name} server with command: {' '.join(command)}")
    process = subprocess.Popen(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    return process

def initservers():
    # Check that npm and npx are installed (install them if needed).
    check_install_node_tools()

    # Determine the base directory (one level up from the current file's directory).
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, 'config-MCP.json')
    config = load_config(config_path)
    print("Loaded configuration: ", config)

    # Launch each MCP server defined in the configuration.
    processes = {}
    for server_name in config.get("mcpServers", {}):
        processes[server_name] = launch_mcp_server(config, server_name)

    # Read and print output from all processes for debugging.
    try:
        while True:
            for server_name, process in processes.items():
                output = process.stdout.readline()
                if output:
                    sys.stdout.write(f"[{server_name}] {output}")
            # Break the loop when all processes have exited.
            if all(process.poll() is not None for process in processes.values()):
                break
    except KeyboardInterrupt:
        print("Terminating all servers...")
        for process in processes.values():
            process.terminate()

if __name__ == "__main__":
    initservers()
