# Complete Guide: Deepseek Setup on Proxmox with Dual RTX 4080 Super

## 1. Proxmox Host Preparation

### 1.1 Enable IOMMU
1. Edit GRUB configuration:
```bash
nano /etc/default/grub
```
2. Add IOMMU support:
```
GRUB_CMDLINE_LINUX_DEFAULT="quiet intel_iommu=on iommu=pt"
```
3. Update GRUB:
```bash
update-grub
```

### 1.2 Load Required Modules
1. Edit modules file:
```bash
nano /etc/modules
```
2. Add these lines:
```
vfio
vfio_iommu_type1
vfio_pci
vfio_virqfd
```

### 1.3 Blacklist NVIDIA Drivers on Host
```bash
nano /etc/modprobe.d/blacklist.conf
```
Add:
```
blacklist nouveau
blacklist nvidia
blacklist nvidia_drm
blacklist nvidia_uvm
blacklist nvidia_modeset
```

### 1.4 Configure GPU Passthrough
1. Identify GPU PCI IDs:
```bash
lspci -nn | grep NVIDIA
```
2. Create VFIO configuration:
```bash
nano /etc/modprobe.d/vfio.conf
```
Add (replace with your PCI IDs):
```
options vfio-pci ids=YOUR_GPU1_ID,YOUR_GPU2_ID
```

### 1.5 Update System
```bash
update-initramfs -u
reboot
```

## 2. Create Ubuntu Server VM

### 2.1 VM Basic Setup
1. In Proxmox web interface:
   - Create VM
   - Choose Ubuntu Server ISO
   - Set VM ID (note this number)
   - Allocate at least 64GB RAM
   - 16 CPU cores minimum
   - 250GB+ storage
   - Q35 machine type
   - OVMF (UEFI) bios

### 2.2 Configure VM for GPU Passthrough
1. Edit VM configuration:
```bash
nano /etc/pve/qemu-server/YOUR_VM_ID.conf
```
2. Add these lines:
```
args: -cpu 'host,+kvm_pv_unhalt,+kvm_pv_eoi,hv_vendor_id=NV43FIX,hv_spinlocks=0x1fff'
machine: q35
cpu: host,hidden=1,flags=+pcid
hostpci0: YOUR_GPU1_ID
hostpci1: YOUR_GPU2_ID
```

## 3. Ubuntu Server Configuration

### 3.1 Basic System Setup
```bash
apt update && apt upgrade -y
apt install -y build-essential cmake git python3-pip python3-dev
```

### 3.2 Install NVIDIA Drivers
```bash
apt install -y nvidia-driver-535 nvidia-utils-535
```

### 3.3 Install CUDA
```bash
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run
```

### 3.4 Set Environment Variables
```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

## 4. Deepseek Installation

### 4.1 Create Python Environment
```bash
apt install -y python3.10-venv
python3 -m venv deepseek-env
source deepseek-env/bin/activate
```

### 4.2 Install Dependencies
```bash
pip install torch torchvision torchaudio
pip install transformers accelerate
pip install deepseek-ai
```

### 4.3 Configure Multi-GPU Setup
Create configuration file:
```bash
nano ~/deepseek-config.json
```
Add:
```json
{
    "gpu_config": {
        "device_ids": [0, 1],
        "memory_allocation": {
            "gpu0_memory_fraction": 0.95,
            "gpu1_memory_fraction": 0.95
        },
        "optimization": {
            "mixed_precision": true,
            "gradient_checkpointing": true,
            "gradient_accumulation_steps": 4
        }
    },
    "training": {
        "batch_size_per_gpu": 8,
        "learning_rate": 2e-5,
        "weight_decay": 0.01
    }
}
```

### 4.4 Create Monitoring Tools
```bash
apt install -y nvtop htop
```

Create GPU monitoring script:
```bash
nano ~/monitor-gpus.sh
```
Add:
```bash
#!/bin/bash
while true; do
    clear
    echo "Deepseek GPU Monitoring"
    echo "----------------------"
    nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader
    sleep 5
done
```

Make executable:
```bash
chmod +x ~/monitor-gpus.sh
```

## 5. Verification Steps

### 5.1 Verify GPU Detection
```bash
nvidia-smi
```

### 5.2 Verify CUDA
```bash
nvcc --version
```

### 5.3 Verify PyTorch GPU Access
```python
python3 -c "import torch; print('GPU available:', torch.cuda.is_available()); print('Number of GPUs:', torch.cuda.device_count()); print('GPU devices:', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
```

## 6. Performance Optimization

### 6.1 Configure CPU Governor
```bash
apt install -y cpufrequtils
echo 'GOVERNOR="performance"' | sudo tee /etc/default/cpufrequtils
systemctl restart cpufrequtils
```

### 6.2 Configure Swap
```bash
fallocate -l 32G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
echo '/swapfile none swap sw 0 0' >> /etc/fstab
```

### 6.3 Configure System Limits
```bash
echo '*       soft    nofile  65535' >> /etc/security/limits.conf
echo '*       hard    nofile  65535' >> /etc/security/limits.conf
```

## 7. Troubleshooting

Common issues and solutions:

1. If GPUs are not detected:
   ```bash
   lspci -k | grep -A 2 NVIDIA
   ```
   Check if vfio-pci is the driver in use

2. If CUDA is not found:
   ```bash
   echo $PATH
   echo $LD_LIBRARY_PATH
   ```
   Verify CUDA paths are correct

3. For memory errors:
   ```bash
   free -h
   ```
   Check available system memory
