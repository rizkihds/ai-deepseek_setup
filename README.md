# DeepSeek LLM Self-Hosting on Ubuntu Server

## 1. Hardware Requirements

- **GPU**: RTX 3090/4090, A100, H100 (24GB+ VRAM recommended)
- **RAM**: 32GB+ (64GB recommended for large models)
- **Storage**: 100GB+ SSD (for model weights & dependencies)
- **CPU**: Multi-core (for better inference performance)

## 2. Setting Up Proxmox for Ubuntu Server

Proxmox is a virtualization platform that allows you to create and manage virtual machines (VMs). If you want to run Ubuntu Server on Proxmox, follow these steps:

### Install Proxmox

1. Download the **Proxmox VE ISO** from [Proxmox official site](https://www.proxmox.com/en/downloads).
2. Create a bootable USB using `balenaEtcher` or `Rufus`.
3. Boot from the USB and install Proxmox on your machine.
4. Follow the installation wizard and configure the network settings.

### Create an Ubuntu Server VM

1. Log in to the **Proxmox Web Interface** (`https://your-proxmox-ip:8006`).
2. Click **Create VM** and enter a VM name.
3. Choose **Ubuntu Server ISO** under the CD/DVD drive (upload it via **Proxmox ISO Storage**).
4. Set **CPU Cores** (4+ recommended) and **Memory** (32GB+ recommended).
5. Allocate **Disk Storage** (100GB+ SSD recommended).
6. Configure **Network** (use `virtio` for best performance).
7. Start the VM and proceed with the **Ubuntu Server installation**.

### Install Ubuntu Server Inside Proxmox VM

Once the VM is running, follow these steps:

1. Boot the VM using the **Ubuntu Server ISO**.

2. Follow the installation process and set up the **username, password, and network settings**.

3. Once installed, update the system:

   ```bash
   sudo apt update && sudo apt upgrade -y
   ```

4. Install **QEMU Guest Agent** for better VM integration:

   ```bash
   sudo apt install qemu-guest-agent -y
   systemctl enable qemu-guest-agent --now
   ```

Now your Ubuntu Server is running inside Proxmox, ready for **DeepSeek LLM installation**.

## 3. System Preparation (Ubuntu Server)

### Update & Install Essential Packages

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install git python3 python3-pip wget curl -y
```

### Install NVIDIA Drivers (if using GPU)

```bash
sudo apt install nvidia-driver-535  # Change version if needed
reboot  # Restart the server
```

### Verify GPU & CUDA Compatibility

```bash
nvidia-smi  # Check GPU status
```

### Use tmux for Running Long Processes (Optional)

```bash
sudo apt install tmux
```

- Start session: `tmux new -s deepseek`
- Detach session: `Ctrl+B, D`
- Reattach session: `tmux attach -t deepseek`

## 4. Software Setup

### Install Python 3.8+

```bash
sudo apt install python3 python3-pip -y
```

### Install CUDA & cuDNN (for NVIDIA GPU acceleration)

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install cuda -y
```

### Install PyTorch (with CUDA support)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 5. DeepSeek LLM Installation & Setup

### Clone DeepSeek Repository

```bash
git clone https://github.com/DeepSeek-LM/DeepSeek-LLM.git /opt/deepseek
cd /opt/deepseek
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run DeepSeek LLM

```bash
python server.py --model deepseek-llm
```

## 6. Backup & Recovery Plan

### Setting Up Backup with `rclone`

1. Install `rclone`:

   ```bash
   curl https://rclone.org/install.sh | sudo bash
   ```

2. Configure `rclone` for Digital Ocean:

   ```bash
   rclone config
   ```

   - Choose **New remote**
   - Enter a name (e.g., `do_spaces`)
   - Select `s3` as the storage type
   - Set provider to `Digital Ocean`
   - Enter Access Key and Secret Key
   - Set region (`nyc3` or your region)
   - Save configuration

3. Sync files to Digital Ocean:

   ```bash
   rclone sync /opt/deepseek do_spaces:your-bucket-name/deepseek-backup
   ```

## 7. Database Storage & Analytics

DeepSeek LLM can connect to a database for storing logs, queries, and responses for analytics.

### Configuring MariaDB Connection

1. Install MariaDB:

   ```bash
   sudo apt install mariadb-server -y
   sudo systemctl enable mariadb --now
   ```

2. Secure MariaDB installation:

   ```bash
   sudo mysql_secure_installation
   ```

3. Create a database for DeepSeek:

   ```sql
   CREATE DATABASE deepseek_db;
   CREATE USER 'deepseek_user'@'localhost' IDENTIFIED BY 'yourpassword';
   GRANT ALL PRIVILEGES ON deepseek_db.* TO 'deepseek_user'@'localhost';
   FLUSH PRIVILEGES;
   ```

4. Configure DeepSeek to use the database:

   - Edit `/opt/deepseek/config/db_connection.py`
   - Update the database connection settings:

   ```python
   DB_HOST = 'localhost'
   DB_USER = 'deepseek_user'
   DB_PASSWORD = 'yourpassword'
   DB_NAME = 'deepseek_db'
   ```

With these steps, you now have DeepSeek LLM running on your **Ubuntu Server** inside **Proxmox**, with **Oobabooga Web UI** for easy interaction. 🚀

Additionally, you can store and analyze model interactions by connecting DeepSeek to your database, including **MariaDB**. 🔍📊

Now, you also have a **backup & recovery plan** to ensure your setup remains secure and recoverable, with automated uploads to **Digital Ocean Spaces** using `rclone`. 🔄☁️

