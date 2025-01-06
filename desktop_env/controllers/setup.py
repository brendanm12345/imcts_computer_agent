"""This version is drastically simplified from the OSWorld version. """

import json
import logging
import os
import time
from typing import Any, Union, List, Dict

import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder

logger = logging.getLogger("desktopenv.setup")


class SetupController:
    def __init__(self, vm_ip: str, server_port: int = 5000, chromium_port: int = 9222, vlc_port: int = 8080, cache_dir: str = "cache"):
        self.vm_ip = vm_ip
        self.server_port = server_port
        self.http_server = f"http://{vm_ip}:{server_port}"
        self.cache_dir = cache_dir

    def reset_cache_dir(self, cache_dir: str):
        self.cache_dir = cache_dir

    def setup(self, config: List[Dict[str, Any]]):
        """Process setup configurations for the environment."""
        for cfg in config:
            config_type = cfg["type"]
            parameters = cfg["parameters"]

            setup_function = "_{:}_setup".format(config_type)
            if hasattr(self, setup_function):
                getattr(self, setup_function)(**parameters)
                logger.info("SETUP: %s(%s)", setup_function, str(parameters))

    def _download_setup(self, files: List[Dict[str, str]]):
        """Download files to the VM."""
        for f in files:
            url = f["url"]
            path = f["path"]

            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()

                # Upload to VM
                form = MultipartEncoder({
                    "file_path": path,
                    "file_data": (os.path.basename(path), response.raw)
                })
                headers = {"Content-Type": form.content_type}

                upload_response = requests.post(
                    f"{self.http_server}/setup/upload",
                    headers=headers,
                    data=form
                )

                if upload_response.status_code == 200:
                    logger.info("File downloaded and uploaded successfully")
                else:
                    logger.error(
                        "Failed to upload file. Status code: %s", upload_response.status_code)

            except Exception as e:
                logger.error(
                    "Failed to handle file: %s. Error: %s", url, str(e))

    def _upload_file_setup(self, files: List[Dict[str, str]]):
        """Upload local files to the VM."""
        for f in files:
            local_path = f["local_path"]
            path = f["path"]

            if not os.path.exists(local_path):
                logger.error(f"Invalid local path ({local_path})")
                continue

            try:
                form = MultipartEncoder({
                    "file_path": path,
                    "file_data": (os.path.basename(path), open(local_path, "rb"))
                })
                headers = {"Content-Type": form.content_type}

                response = requests.post(
                    f"{self.http_server}/setup/upload",
                    headers=headers,
                    data=form
                )

                if response.status_code == 200:
                    logger.info("File uploaded successfully")
                else:
                    logger.error(
                        "Failed to upload file. Status code: %s", response.text)

            except Exception as e:
                logger.error("Failed to upload file: %s. Error: %s",
                             local_path, str(e))

    def _execute_setup(self, command: List[str], shell: bool = False):
        """Execute command on the VM."""
        try:
            payload = json.dumps({"command": command, "shell": shell})
            headers = {"Content-Type": "application/json"}

            response = requests.post(
                f"{self.http_server}/setup/execute",
                headers=headers,
                data=payload
            )

            if response.status_code == 200:
                logger.info("Command executed successfully: %s", command)
            else:
                logger.error(
                    "Failed to execute command. Status code: %s", response.status_code)

        except Exception as e:
            logger.error(
                "Failed to execute command: %s. Error: %s", command, str(e))

    def _command_setup(self, command: List[str], **kwargs):
        """Execute a command on the VM."""
        self._execute_setup(command, **kwargs)

    def _sleep_setup(self, seconds: float):
        """Sleep for specified duration."""
        time.sleep(seconds)
