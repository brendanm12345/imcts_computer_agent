from abc import ABC, abstractmethod


class Provider(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def start_emulator(self, path_to_vm: str, headless: bool, os_type: str):
        pass

    @abstractmethod
    def get_ip_address(self, path_to_vm: str) -> str:
        pass

    @abstractmethod
    def revert_to_snapshot(self, path_to_vm: str, snapshot_name: str) -> str:
        pass

    @abstractmethod
    def stop_emulator(self, path_to_vm: str):
        pass


class VMManager(ABC):
    @abstractmethod
    def get_vm_path(self, os_type: str) -> str:
        pass
