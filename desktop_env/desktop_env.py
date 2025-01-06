from __future__ import annotations

import logging
import os
import time
from typing import Dict, Any, Optional, Tuple, List

import gymnasium as gym

from desktop_env.controllers.python import PythonController
from desktop_env.controllers.setup import SetupController
from desktop_env.providers import create_vm_manager_and_provider

logger = logging.getLogger("desktopenv.env")


class DesktopEnv(gym.Env):
    """
    DesktopEnv with OpenAI Gym interface. Provides a desktop environment for automation tasks.
    """

    def __init__(
            self,
            path_to_vm: str = None,
            snapshot_name: str = "init_state",
            cache_dir: str = "cache",
            screen_size: Tuple[int] = (1920, 1080),
            headless: bool = False,
            os_type: str = "Ubuntu",
    ):
        # Initialize VM manager and virtualization provider
        self.region = None
        self.provider_name = "vmware"

        # Default ports
        self.server_port = 5000
        self.chromium_port = 9222
        self.vnc_port = 8006
        self.vlc_port = 8080

        self.manager, self.provider = create_vm_manager_and_provider(
            self.provider_name, self.region)
        self.os_type = os_type

        # Initialize environment variables
        if path_to_vm:
            self.path_to_vm = os.path.abspath(
                os.path.expandvars(os.path.expanduser(path_to_vm)))
        else:
            self.path_to_vm = self.manager.get_vm_path(
                self.os_type)

        self.snapshot_name = snapshot_name
        self.cache_dir_base = cache_dir
        self.headless = headless

        # Initialize emulator
        logger.info("Initializing...")
        self._start_emulator()

        self.instruction = None
        self.action_space = "pyautogui"

        # Episode tracking
        self._traj_no = -1
        self._step_no = 0
        self.action_history: List[Dict[str, any]] = []

    def _start_emulator(self):
        # Power on the virtual machine
        self.provider.start_emulator(
            self.path_to_vm, self.headless, self.os_type)

        # Get the ip from the virtual machine and setup controller
        vm_ip_ports = self.provider.get_ip_address(self.path_to_vm).split(':')
        self.vm_ip = vm_ip_ports[0]
        if len(vm_ip_ports) > 1:
            self.server_port = int(vm_ip_ports[1])
            self.chromium_port = int(vm_ip_ports[2])
            self.vnc_port = int(vm_ip_ports[3])
            self.vlc_port = int(vm_ip_ports[4])

        self.controller = PythonController(
            vm_ip=self.vm_ip, server_port=self.server_port)
        self.setup_controller = SetupController(
            vm_ip=self.vm_ip,
            server_port=self.server_port,
            chromium_port=self.chromium_port,
            vlc_port=self.vlc_port,
            cache_dir=self.cache_dir_base
        )

    def _revert_to_snapshot(self):
        # Revert to snapshot and handle path updates
        path_to_vm = self.provider.revert_to_snapshot(
            self.path_to_vm, self.snapshot_name)
        if path_to_vm and not path_to_vm == self.path_to_vm:
            self.manager.delete_vm(self.path_to_vm, self.region)
            self.manager.add_vm(path_to_vm, self.region)
            self.manager.occupy_vm(path_to_vm, os.getpid(), self.region)
            self.path_to_vm = path_to_vm

    def close(self):
        # Close the virtual machine
        self.provider.stop_emulator(self.path_to_vm)

    def reset(self, task_config: Optional[Dict[str, Any]] = None, seed=None, options=None) -> Dict[str, Any]:
        logger.info("Resetting environment...")
        self._traj_no += 1
        self._step_no = 0
        self.action_history.clear()

        logger.info(f"Reverting to snapshot {self.snapshot_name}...")
        self._revert_to_snapshot()
        logger.info("Starting emulator...")
        self._start_emulator()
        logger.info("Emulator started.")

        if task_config is not None:
            self._set_task_info(task_config)
            self.setup_controller.reset_cache_dir(self.cache_dir)
            logger.info("Setting up environment...")
            self.setup_controller.setup(self.config)
            logger.info("Environment setup complete.")

        return self._get_obs()

    def _get_obs(self):
        return {
            "screenshot": self.controller.get_screenshot(),
            "instruction": self.instruction
        }

    def _set_task_info(self, task_config: Dict[str, Any]):
        self.task_id = task_config["id"]
        self.cache_dir = os.path.join(self.cache_dir_base, self.task_id)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.instruction = task_config["instruction"]
        self.config = task_config.get("config", [])

    def step(self, action, pause=0.5):
        self._step_no += 1
        self.action_history.append(action)

        # Default values
        reward = 0
        done = False
        info = {}

        # Handle special actions
        if action in ['WAIT', 'FAIL', 'DONE']:
            if action == 'WAIT':
                time.sleep(pause)
            elif action == 'FAIL':
                done = True
                info = {"fail": True}
            elif action == 'DONE':
                done = True
                info = {"done": True}
        else:
            # Execute PyAutoGUI command
            self.controller.execute_python_command(action)

        return self._get_obs(), reward, done, info

    def render(self, mode='rgb_array'):
        if mode == 'rgb_array':
            return self.controller.get_screenshot()
        else:
            raise ValueError('Unsupported render mode: {}'.format(mode))
