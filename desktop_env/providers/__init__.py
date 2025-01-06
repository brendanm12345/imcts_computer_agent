from desktop_env.providers.base import VMManager, Provider
from desktop_env.providers.vmware.manager import VMwareVMManager
from desktop_env.providers.vmware.provider import VMwareProvider


def create_vm_manager_and_provider(provider_name: str = "vmware", region: str = None):
    """
    Creates VMware manager and provider instances.
    Default provider is VMware and region is not used but kept for compatibility.
    """
    return VMwareVMManager(), VMwareProvider()
