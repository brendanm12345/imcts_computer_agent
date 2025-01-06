# Imaginary-MCTS: Autonomous Computer Use Agents with Enhanced Online Planning

To improve computer agent reasoning and boost task completion rates, we propose an online planning algorithm inspired by Monte Carlo Tree Search in which the agent is prompted to imagine the state changes associated with each candidate action, effectively serving as trajectory rollouts that avoid actually executing and candidate actions and then backtracking which is challenging, slow, and error-prone. The agent then scores each candidate action based on its imagined rollout and proceeds with the highest-scoring action. For more information the algorithm and its results please view our [write-up](https://github.com/user-attachments/files/18131847/Imaginary-MCTS-Write-Up.pdf). The virtualization, evaluation_examples, and baseline agent in this repository adapt or draw code from the [OSWorld](https://github.com/xlang-ai/OSWorld) repository. This repository forgoes the evaluation scripts and variety of supported virtualization and foundation model providers present in the OSWorld repo in favor of constructing a maximally lightweight computer agent that's easy to understand and experiment with.

OSWorld is a popular computer agent benchmark.

## Installation

1. Clone the repo and install packages

```bash
# Clone the OSWorld repository
git clone https://github.com/brendanm12345/imcts_computer_agent

# Change directory into the cloned repository
cd imcts_computer_agent

# Optional: Create a Conda environment for OSWorld
# conda create -n imcts
# conda activate imcts

# Install required dependencies
pip install -r requirements.txt
```

2. (from OSWorld) Install [VMware Workstation Pro](https://www.vmware.com/products/workstation-pro/workstation-pro-evaluation.html) (for systems with Apple Chips, you should install [VMware Fusion](https://support.broadcom.com/group/ecx/productdownloads?subfamily=VMware+Fusion)) and configure the `vmrun` command. The installation process can refer to [How to install VMware Worksation Pro](desktop_env/providers/vmware/INSTALL_VMWARE.md). Verify the successful installation by running the following:

```bash
vmrun -T ws list
```

If the installation along with the environment variable set is successful, you will see the message showing the current running virtual machines.

## Verify Correct VM Setup

To verify that your virtualization has been done correctly, run

```bash
python3 quickstart.py
```

If things are working correctly you should see:

- The VMWare Fusion application open showing a desktop screen
- A right-click get executed in the middle of the desktop screen, showing the Ubuntu pop-up menu like the below image:
<img width="1500" alt="Screenshot 2025-01-05 at 6 32 50 PM" src="https://github.com/user-attachments/assets/adc8ebe3-da68-483c-951c-8f8c4961ee26" />


If you see a desktop screen prompting you for a password, enter `password` as the password and run the `quickstart.py` script again

Now that we have a VM to use, let's run the agent!

## Run the Agent

1. Set **ANTHROPIC_API_KEY** environment variable with your API key

```bash
export ANTHROPIC_API_KEY='changeme'
```

2. Run the baseline agent

```bash
python3 run.py --path_to_vm vmware_vm_data/Ubuntu0/Ubuntu0.vmx --model claude-3-5-sonnet-latest --result_dir ./results
```

3. Run the IMCTS agent

```bash
python3 run.py --path_to_vm vmware_vm_data/Ubuntu0/Ubuntu0.vmx --model claude-3-5-sonnet-latest --result_dir ./results --imcts
```

Note: you may need to update the `path_to_vm`

The results, which include screenshots, actions, and video recordings of the agent's task completion, will be saved in the `./results` directory in this case. The logs containing the agents reasoning wil be saved in the `.logs` directory

---

Stanford University, CS 238 Final Project. Authors: Brendan McLaughlin (BS'24, MS'25), Michael Maffezzoli (BS'23, MS'24), under the guidance of Professor Mykel J. Kochenderfer. Grade: A+
