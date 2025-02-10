# System FLOW
```
System Setup - Flow
     │
     ├──► Install Proxmox
     │       ├──► Create Ubuntu VM
     │       ├──► Install Ubuntu Server
     │       └──► Configure System (Updates, Drivers, tmux)
     │
     ├──► Install Required Software
     │       ├──► Python, CUDA, PyTorch
     │       ├──► NVIDIA Drivers (if GPU)
     │       └──► MariaDB (for logging & analytics)
     │
     ├──► DeepSeek LLM Installation
     │       ├──► Clone Repository
     │       ├──► Install Dependencies
     │       └──► Start DeepSeek LLM Server
     │
     ├──► Oobabooga Web UI Setup
     │       ├──► Install Web UI
     │       ├──► Connect to DeepSeek LLM
     │       └──► Serve Web UI for User Interaction
     │
     ├──► Request Processing Flow
     │       ├──► User Sends Request (Web UI)
     │       ├──► DeepSeek LLM Processes Query
     │       ├──► (Optional) Log Request to MariaDB
     │       ├──► Generate Response
     │       └──► Send Response Back to User
     │
     ├──► Backup & Recovery
     │       ├──► Install & Configure `rclone`
     │       ├──► Connect to Digital Ocean Spaces
     │       ├──► Sync Model & Data (`/opt/deepseek`)
     │       ├──► Verify Backup
     │       └──► Restore from Backup (if needed)
     │
     └──► Analytics & Database Storage
             ├──► Store User Queries in MariaDB
             ├──► Analyze Query Trends
             ├──► Optimize Model Based on Data
             └──► Retrieve Stored Logs for Review

```
# Setting Up Proxmox for Ubuntu Server

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

# DeepSeek LLM Self-Hosting on Ubuntu Server

## 1. Hardware Requirements

- **GPU**: RTX 3090/4090, A100, H100 (24GB+ VRAM recommended)
- **RAM**: 32GB+ (64GB recommended for large models)
- **Storage**: 100GB+ SSD (for model weights & dependencies)
- **CPU**: Multi-core (for better inference performance)

## 2. System Preparation (Ubuntu Server)

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

## 3. Software Setup

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

## 4. DeepSeek LLM Installation & Setup

### Set Up Installation Directory

```bash
mkdir -p /home/ubuntu/deepseek
cd /home/ubuntu/deepseek
```

### Download DeepSeek Model

- Visit: [Hugging Face](https://huggingface.co/deepseek-ai)
- Download the model files and place them in the `models/` directory:

```bash
mkdir -p /home/ubuntu/deepseek/models
cd /home/ubuntu/deepseek/models
# Download model weights manually or use huggingface-cli
```

### Install Additional Dependencies

```bash
pip install transformers accelerate sentencepiece
```

### Run DeepSeek Model with Python (Standalone Test)

```bash
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-llm")
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-llm")

input_text = "Hello, how are you?"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0]))
```

### DeepSeek Data Storage Locations

- **Installation Directory:** `/home/ubuntu/deepseek/`
- **Model Weights & Checkpoints** (if downloaded via Hugging Face):
  ```bash
  ~/.cache/huggingface/hub/
  ```
- **Oobabooga Web UI Model Directory:**
  ```bash
  text-generation-webui/models/
  ```
- **Temporary Cache & Logs:**
  ```bash
  ~/.cache/huggingface/
  ~/.cache/torch/
  ```
- **Custom Model Path:** If specified manually, models will be stored in the chosen directory.

## 5. Connecting DeepSeek to a Database for Analytics

### Install Database Connector

- **PostgreSQL:**
  ```bash
  pip install psycopg2
  ```
- **MySQL/MariaDB:**
  ```bash
  pip install mysql-connector-python
  ```
- **MongoDB:**
  ```bash
  pip install pymongo
  ```

### Store Model Responses in MariaDB

Create a new script file:

#### **File Name: ****\`\`****)**

```python
import mysql.connector

def connect_db():
    conn = mysql.connector.connect(
        host="localhost",
        user="your_user",
        password="your_password",
        database="your_db"
    )
    return conn
```

#### **File Name: ****\`\`****)**

```python
from db_connection import connect_db

def store_response(user_input, model_response):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO analytics (user_input, model_response) VALUES (%s, %s)", (user_input, model_response))
    conn.commit()
    cursor.close()
    conn.close()
```

## 6. Oobabooga Web UI Installation

### Clone Repository

```bash
git clone https://github.com/oobabooga/text-generation-webui.git
cd text-generation-webui
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Oobabooga Web UI with DeepSeek

```bash
python server.py --model deepseek-llm
```

## 7. Backup & Upload to Digital Ocean Spaces

### Backup Model Files, Database, and Configurations

```bash
tar -czvf deepseek_backup_$(date +%F).tar.gz /home/ubuntu/deepseek/ ~/.cache/huggingface/hub/ text-generation-webui/models/ ~/deepseek_db_backup.sql
```

### Setup & Configure s3cmd for Digital Ocean Spaces

```bash
sudo apt install s3cmd
s3cmd --configure
```

Follow the prompts and enter your Digital Ocean Spaces Access & Secret keys.

### Upload Backup to Digital Ocean Spaces using s3cmd

```bash
s3cmd put deepseek_backup_$(date +%F).tar.gz s3://your-bucket-name/
```

### Setup & Configure rclone for Digital Ocean Spaces

```bash
sudo apt install rclone
rclone config
```

Follow the prompts to add a new remote for Digital Ocean Spaces, choosing **S3-compatible storage** and entering your credentials.

### Upload Backup to Digital Ocean Spaces using rclone

```bash
rclone copy deepseek_backup_$(date +%F).tar.gz your-remote-name:your-bucket-name
```

## 8. Conclusion

With these steps, you now have DeepSeek LLM running on your **Ubuntu Server** with **Oobabooga Web UI** for easy interaction. 🚀

Additionally, you can store and analyze model interactions by connecting DeepSeek to your database, including **MariaDB**. 🔍📊

Your database connections are now modularized in `db_connection.py` and `store_response.py`, making it easier to manage and secure credentials.

Now, you also have a **backup & recovery plan** to ensure your setup remains secure and recoverable, along with automated upload to **Digital Ocean Spaces**. 🔄☁️

