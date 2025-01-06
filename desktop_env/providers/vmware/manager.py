import logging
import os
import platform
import requests
import subprocess
from tqdm import tqdm

from desktop_env.providers.base import VMManager

logger = logging.getLogger("desktopenv.providers.vmware.VMwareVMManager")
logger.setLevel(logging.INFO)

VMS_DIR = "./vmware_vm_data"

UBUNTU_ARM_URL = "https://huggingface.co/datasets/xlangai/ubuntu_osworld/resolve/main/Ubuntu-arm.zip"
UBUNTU_X86_URL = "https://huggingface.co/datasets/xlangai/ubuntu_osworld/resolve/main/Ubuntu-x86.zip"


class VMwareVMManager(VMManager):
    def get_vm_path(self, os_type: str) -> str:
        """Get or create a VM path."""
        os.makedirs(VMS_DIR, exist_ok=True)

        # Get appropriate URL based on system architecture
        if os_type == "Ubuntu":
            if platform.system() == 'Darwin':
                url = UBUNTU_ARM_URL
            elif platform.machine().lower() in ['amd64', 'x86_64']:
                url = UBUNTU_X86_URL
            else:
                raise Exception("Unsupported platform or architecture")
        else:
            raise Exception("Only Ubuntu is supported")

        vm_name = "Ubuntu0"  # Simplified to always use a single VM
        vm_path = os.path.join(VMS_DIR, vm_name, vm_name + ".vmx")

        if not os.path.exists(vm_path):
            self._download_and_setup_vm(url, vm_name)

        return vm_path

    def _download_and_setup_vm(self, url: str, vm_name: str):
        """Download and set up the VM."""
        downloaded_file = os.path.join(VMS_DIR, url.split('/')[-1])

        # Download VM image
        logger.info("Downloading VM image...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(downloaded_file, 'wb') as file, tqdm(
                desc="Progress",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
                ascii=True
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)

        # Extract VM
        logger.info("Extracting VM files...")
        import zipfile
        with zipfile.ZipFile(downloaded_file, 'r') as zip_ref:
            zip_ref.extractall(os.path.join(VMS_DIR, vm_name))

        # Update VM configuration
        logger.info("Setting up VM...")
        old_vmx = os.path.join(VMS_DIR, vm_name, "Ubuntu.vmx")
        new_vmx = os.path.join(VMS_DIR, vm_name, f"{vm_name}.vmx")

        if os.path.exists(old_vmx):
            os.rename(old_vmx, new_vmx)

        # Remove downloaded zip
        os.remove(downloaded_file)
