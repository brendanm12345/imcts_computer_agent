import logging
import os
import platform
import subprocess
import time

from desktop_env.providers.base import Provider

logger = logging.getLogger("desktopenv.providers.vmware.VMwareProvider")
logger.setLevel(logging.INFO)

WAIT_TIME = 3


def get_vmrun_type():
    if platform.system() in ['Windows', 'Linux']:
        return ['-T', 'ws']
    elif platform.system() == 'Darwin':
        return ['-T', 'fusion']
    else:
        raise Exception("Unsupported operating system")


class VMwareProvider(Provider):
    @staticmethod
    def _execute_command(command: list, return_output=False):
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8"
        )

        if return_output:
            output = process.communicate()[0].strip()
            return output

    def start_emulator(self, path_to_vm: str, headless: bool, os_type: str):
        logger.info("Starting VMware VM...")

        while True:
            try:
                output = subprocess.check_output(["vmrun"] + get_vmrun_type() + ["list"],
                                                 stderr=subprocess.STDOUT)
                output = output.decode().splitlines()
                normalized_path = os.path.abspath(os.path.normpath(path_to_vm))

                if any(os.path.abspath(os.path.normpath(line)) == normalized_path for line in output):
                    logger.info("VM is running.")
                    break

                logger.info("Starting VM...")
                command = ["vmrun"] + get_vmrun_type() + ["start", path_to_vm]
                if headless:
                    command.append("nogui")
                self._execute_command(command)
                time.sleep(WAIT_TIME)

            except subprocess.CalledProcessError as e:
                logger.error(f"Error executing command: {
                             e.output.decode().strip()}")

    def get_ip_address(self, path_to_vm: str) -> str:
        logger.info("Getting VMware VM IP address...")
        while True:
            try:
                output = self._execute_command(
                    ["vmrun"] + get_vmrun_type() + ["getGuestIPAddress",
                                                    path_to_vm, "-wait"],
                    return_output=True
                )
                logger.info(f"VMware VM IP address: {output}")
                return output
            except Exception as e:
                logger.error(e)
                time.sleep(WAIT_TIME)
                logger.info("Retrying to get VMware VM IP address...")

    def revert_to_snapshot(self, path_to_vm: str, snapshot_name: str):
        logger.info(f"Reverting VMware VM to snapshot: {snapshot_name}...")
        self._execute_command(
            ["vmrun"] + get_vmrun_type() + ["revertToSnapshot", path_to_vm, snapshot_name])
        time.sleep(WAIT_TIME)
        return path_to_vm

    def stop_emulator(self, path_to_vm: str):
        logger.info("Stopping VMware VM...")
        self._execute_command(
            ["vmrun"] + get_vmrun_type() + ["stop", path_to_vm])
        time.sleep(WAIT_TIME)
