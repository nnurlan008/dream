# DREAM: Device-Driven Efficient Access to Virtual Memory

Sorry for delay. The README will be updated within the week until June 7.



**This is the open-source implementation of the DREAM system (ICS 2025). Contributions to the codebase are most welcome**

Abstract
-------------------------------------------------------------------------------
Graphics Processing Units (GPUs) excel at high-performance computing tasks, including multimedia rendering, cryptomining, deep learning, and natural language processing, due to their massive parallelism and high memory bandwidth. However, the growing size of models and datasets in these domains increasingly exceeds the memory capacity of a single GPU, resulting in significant performance overheads. To mitigate this issue, developers are often forced to partition data and manually manage transfers between GPU and host memory—a labor-intensive approach that becomes impractical for workloads with irregular memory access patterns, such as deep learning, recommendation systems, and graph processing. Programming abstractions like Unified Virtual Memory (UVM) simplify development by offering a unified memory space across the system and handling data transfers automatically. Unfortunately, UVM introduces substantial overhead due to frequent OS involvement and inefficient data movement, particularly when GPU memory is oversubscribed.

This paper presents DREAM, a GPU memory management system that leverages an RDMA-capable network device to implement a programmer-agnostic lightweight virtual memory system, eliminating CPU/OS involvement. DREAM supports on-demand page migration for GPU applications by delegating memory management and page migration tasks to GPU threads. Since current CPU architectures do not support GPU-initiated memory management, DREAM uses a network interface card to enable efficient, transparent page migration. By offloading memory management to the GPU, DREAM achieves up to 4x higher performance than UVM for latency-sensitive applications while maintaining user-friendly programming abstractions that eliminate the need for manual memory management.

Hardware/System Requirements
-------------------------------------------------------------------------------
This code base requires specific type of hardware and specific system configuration to be functional and performant.

### Hardware Requirements ###
* A x86 system supporting PCIe P2P
* NIC: Preferrably Mellanox Connectx
* A NVIDIA Tesla/Datacenter grade GPU that is from the Volta or newer generation. A Tesla V100/A100/H100 fit both of these requirements
  * A Tesla grade GPU is needed as it can expose all of its memory for P2P accesses over PCIe. (NVIDIA Tesla T4 does not work as it only provides 256M of BAR space)
  * A Volta or newer generation of GPU is needed as we rely on memory synchronization primitives only supported since Volta.
We have built our software prototype on Cloudlab r7525 nodes at Clemson. The profile has been provided in the repo.

### System Configurations ###
* As mentioned above, `Above 4G Decoding` needs to be ENABLED in the BIOS
* The system's IOMMU should be disabled for ease of debugging.
  * In Intel Systems, this requires disabling `Vt-d` in the BIOS
  * In AMD Systems, this requires disabling `IOMMU` in the BIOS
* The `iommu` support in Linux must be disabled too, which can be checked and disabled following the instructions [below](#disable-iommu-in-linux).
* In the system's BIOS, `ACS` must be disabled if the option is available
* Preferrably new Linux kernel; 5.x
  
#### If you plan to use the Cloudlab profile, the following commands will setup the system

### Commands for System Setup ###

```
sudo apt-get update
sudo apt-get install nvidia-driver-535
sudo apt-get update && sudo apt-get install -y cmake cython3 dh-python libsystemd-dev libudev-dev pandoc python3-docutils valgrind
```

### MLNX-OFED Driver Installation ###
We use MLNX-OFED-23.07-0.5.1.2 driver for the DREAM software prototype. However, any driver that matches the system requirements can be used.
Installation path for MLNX-OFED driver is assumed to be `$HOME`.

```
sudo wget http://www.mellanox.com/downloads/ofed/MLNX_OFED-23.07-0.5.1.2/MLNX_OFED_LINUX-23.07-0.5.1.2-ubuntu22.04-x86_64.tgz
tar -xvf MLNX_OFED_LINUX-23.07-0.5.1.2-ubuntu22.04-x86_64.tgz
cd MLNX_OFED_LINUX-23.07-0.5.1.2-ubuntu22.04-x86_64/
sudo ./mlnxofedinstall
```

After installation of MLNX-OFED, the interface should be restarted and the system needs to be rebooted:

`sudo /etc/init.d/openibd restart`

After successful installation of MLNX-OFED driver, the custom rdma-core and mlnx-kernel packages which are provided in this repo should be built and installed. 
These packages are modified versions of the original packages and enable (1) allocation of *QP* and *CQ* buffers on GPU memory. For installation, please follow the individual READMEs in rdma_core and mlnx-kernel folders.

### CUDA Installation ###

The following commands will install CUDA-12.2.

```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda-repo-ubuntu2204-12-2-local_12.2.0-535.54.03-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-2-local_12.2.0-535.54.03-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-2-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```

### Enabling Nvidia PeerMem ###
To make sure the NIC can access GPU memory (BAR space), the P2P peermem access should be activated with the following command:

`sudo modprobe nvidia-peermem`


### Compiling Nvidia Driver Kernel Symbols ###
Typically the Nvidia driver kernel sources are installed in the `/usr/src/` directory.
So if the Nvidia driver version is `470.141.03`, then they will be in the `/usr/src/nvidia-470.141.03` directory.
So assuming the driver version is `470.141.03`, to get the kernel symbols you need to do the following commands as the `root` user.

```
$ cd /usr/src/nvidia-470.141.03/
$ sudo make
```

Building the Project
-------------------------------------------------------------------------------
### This part will be updated! ###



# Citations 

If you use DREAM or concepts or derivative codebase of DREAM in your work, please cite the following articles:

```
@article{nazaraliyev2024gpuvm,
  title={GPUVM: GPU-driven Unified Virtual Memory},
  author={Nazaraliyev, Nurlan and Sadredini, Elaheh and Abu-Ghazaleh, Nael},
  journal={arXiv preprint arXiv:2411.05309},
  year={2024}
}

@article{nazaraliyev2025dream,
  title={DREAM: Device-Driven Efficient Access to Virtual Memory},
  author={Nazaraliyev, Nurlan and Sadredini, Elaheh and Abu-Ghazaleh, Nael},
  year={2025}
}
```
