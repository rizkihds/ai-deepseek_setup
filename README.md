# DeepSeek - System FLOW 🚀
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

# DeepSeek Data Storage & Locations  🔍

* **Installation Directory:**`/home/ubuntu/deepseek/`
* **Model Weights & Checkpoints** (if downloaded via Hugging Face):
  ```bash
  ~/.cache/huggingface/hub/
  ```
* **Oobabooga Web UI Model Directory:**
  ```bash
  text-generation-webui/models/
  ```
* **Temporary Cache & Logs:**
  ```bash
  ~/.cache/huggingface/
  ~/.cache/torch/
  ```
* **Custom Model Path:** If specified manually, models will be stored in the chosen directory.


# DeepSeek LLM Self-Hosting on Ubuntu Server Inside Proxmox 🚀

## 1. Hardware Requirements

* **GPU**: RTX 3090/4090, A100, H100 (24GB+ VRAM recommended)
* **RAM**: 32GB+ (64GB recommended for large models)
* **Storage**: 100GB+ SSD (for model weights & dependencies)
* **CPU**: Multi-core (for better inference performance)

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

* Start session: `tmux new -s deepseek`
* Detach session: `Ctrl+B, D`
* Reattach session: `tmux attach -t deepseek`

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

## 6. User Interaction Flow

```
Client Request
     │
     ▼
Oobabooga Web UI
     │
     ▼
DeepSeek LLM Server
     │
     ├──► (Optional) Log request to MariaDB
     │
     ▼
Generate Response
     │
     ▼
Oobabooga Web UI
     │
     ▼
Client Receives Response
```

## Oobabooga Web UI Installation

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

## 7. Backup & Recovery Plan

### Setting Up Backup with `rclone`

```
Backup Process
     │
     ▼
Install & Configure rclone
     │
     ▼
Setup Digital Ocean Spaces
     │
     ▼
Authenticate & Configure rclone
     │
     ▼
Select Backup Directory (/opt/deepseek)
     │
     ▼
Initiate Backup with rclone sync
     │
     ▼
Data Uploaded to Digital Ocean Spaces
     │
     ▼
Verify Backup Integrity
```

### Setting Up Backup with `rclone`

1. Install `rclone`:

   ```bash
   curl https://rclone.org/install.sh | sudo bash
   ```
2. Configure `rclone` for Digital Ocean:

   ```bash
   rclone config
   ```

   * Choose **New remote**
   * Enter a name (e.g., `do_spaces`)
   * Select `s3` as the storage type
   * Set provider to `Digital Ocean`
   * Enter Access Key and Secret Key
   * Set region (`nyc3` or your region)
   * Save configuration
     
4. Backup Model Files, Database, and Configurations

   ```bash
   tar -czvf deepseek_backup_$(date +%F).tar.gz /home/ubuntu/deepseek/ ~/.cache/huggingface/hub/ text-generation-webui/models/ ~/deepseek_db_backup.sql
   ```

5. Copy files to Digital Ocean:

   ```bash
   rclone copy deepseek_backup_$(date +%F).tar.gz your-remote-name:your-bucket-name
   ```

   
## 8. Database Storage & Analytics


### Install Database Connector

* **MySQL/MariaDB:**
  ```bash
  pip install mysql-connector-python
  ```

### Store Model Responses in MariaDB

Create a new script file:

#### **File Name: `db_connection.py` **

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

#### **File Name: `store_response.py` **

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

   * Edit `/opt/deepseek/config/db_connection.py`
   * Update the database connection settings:

   ```python
   DB_HOST = 'localhost'
   DB_USER = 'deepseek_user'
   DB_PASSWORD = 'yourpassword'
   DB_NAME = 'deepseek_db'
   ```
5. Log Queries to MariaDB
   * Modify /opt/deepseek/server.py to log incoming queries.
     ```python
     import mysql.connector
     import datetime
     
     def log_query(user_query, model_response):
         conn = mysql.connector.connect(
             host="localhost",
             user="deepseek_user",
             password="yourpassword",
             database="deepseek_db"
         )
         cursor = conn.cursor()
    
         query = "INSERT INTO query_logs (timestamp, query, response) VALUES (%s, %s, %s)"
         values = (datetime.datetime.now(), user_query, model_response)
         
         cursor.execute(query, values)
         conn.commit()
         conn.close()
     
     # Call log_query(user_query, model_response) inside the request handler
     ```
   * Create a Table for Logging
     Run the following SQL command in MariaDB:
     
     ```sql
     CREATE TABLE query_logs (
         id INT AUTO_INCREMENT PRIMARY KEY,
         timestamp DATETIME,
         query TEXT,
         response TEXT
     );
     ```
   * Analyze Trends Using SQL
     - To analyze query trends:

     ```sql
     SELECT query, COUNT(*) AS count FROM query_logs GROUP BY query ORDER BY count DESC LIMIT 10;
     ```
     - Query Trends Over Time

     ```sql
     SELECT DATE(timestamp) AS date, COUNT(*) AS query_count 
     FROM query_logs GROUP BY date ORDER BY date DESC;
     ```
   
## 9. Full Process Overview

```
System Setup
     │
     ├──► Install Proxmox
     │       ├──► Create Ubuntu VM
     │       ├──► Install Ubuntu Server
     │       └──► Configure System
     │
     ├──► Install Required Software
     │
     ├──► DeepSeek LLM Installation
     │
     ├──► Oobabooga Web UI Setup
     │
     ├──► Request Processing Flow
     │
     ├──► Backup & Recovery
     │
     └──► Analytics & Database Storage
```

With these steps, you now have DeepSeek LLM running on your **Ubuntu Server** inside **Proxmox**, with **Oobabooga Web UI** for easy interaction. 🚀

Additionally, you can store and analyze model interactions by connecting DeepSeek to your database, including **MariaDB**. 🔍📊

Your database connections are now modularized in `db_connection.py` and `store_response.py`, making it easier to manage and secure credentials.

Now, you also have a **backup & recovery plan** to ensure your setup remains secure and recoverable, along with automated upload to **Digital Ocean Spaces**. 🔄☁️

# Optimize DeepSeek LLM based on Data📊
* follow these steps
### 1. **Enable Query Logging & Analysis**

* Ensure that all user queries and model responses are stored in MariaDB.
* Use SQL queries to analyze trends (e.g., most common queries, response times).

### 2. **Fine-Tune the Model**

* Collect frequently asked queries and their responses.
* Train a smaller fine-tuned model using **LoRA (Low-Rank Adaptation)** or **QLoRA** to reduce GPU memory usage.

### 3. **Cache Frequent Queries**

* Implement a caching layer (e.g., **Redis**) to store responses for common queries, reducing inference load.

### 4. **Optimize GPU Usage**

* Enable **TensorRT** or **bitsandbytes** for better performance.
* Use **model quantization** to reduce memory footprint.

### 5. **Database Indexing for Faster Query Analysis**

* Add indexes on frequently queried columns in MariaDB:
```sql
CREATE INDEX idx_query ON query_logs(query);
```

# Use Redis for caching frequent queries in DeepSeek 🔍:
* follow these steps
### 1. Install Redis on Ubuntu Server

```bash
sudo apt update && sudo apt install redis -y
sudo systemctl enable redis --now
```

### 2. Install Redis Python Library

```bash
pip install redis
```

### 3. Modify DeepSeek to Use Redis for Caching

* Edit /opt/deepseek/server.py and update the request processing logic to include caching.
a. Connect to Redis

```python
import redis
import hashlib
import json

# Connect to Redis
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=True)
```

b. Cache Query Responses

* Modify the function that processes user queries:

```python
def process_query(user_query):
# Generate a hash key for the query
query_key = hashlib.sha256(user_query.encode()).hexdigest()

# Check if the query response is cached
cached_response = redis_client.get(query_key)
if cached_response:
   return json.loads(cached_response)  # Return cached response

# If not cached, generate response using DeepSeek
model_response = generate_response(user_query)

# Store response in Redis with an expiration time (e.g., 24 hours)
redis_client.setex(query_key, 86400, json.dumps(model_response))

return model_response
```

### 4. Test Redis Caching

* Run the script and check if responses are cached.
  
```bash
redis-cli KEYS *
```

# Fine-Tuning DeepSeek LLM with LoRA & QLoRA 🔍🚀

## Introduction
Fine-tuning large language models like DeepSeek LLM can be resource-intensive. To optimize memory usage and computational efficiency, **LoRA (Low-Rank Adaptation)** and **QLoRA (Quantized LoRA)** techniques allow fine-tuning with reduced GPU memory requirements.

## 1. Understanding LoRA & QLoRA

### LoRA (Low-Rank Adaptation)
- LoRA introduces **trainable low-rank matrices** that are injected into the model's layers, reducing the number of parameters to update.
- This method enables efficient fine-tuning without modifying the entire model.
- LoRA updates only a small subset of model parameters, making training faster and more memory-efficient.

### QLoRA (Quantized LoRA)
- QLoRA **quantizes** the model (e.g., 4-bit precision) to drastically lower memory usage while preserving performance.
- Uses **NF4 (NormalFloat4) quantization** combined with LoRA adapters.
- Reduces VRAM consumption, making fine-tuning possible on consumer-grade GPUs.
- Maintains nearly full model performance while significantly reducing computational requirements.

### Comparison of LoRA vs. QLoRA
| Feature          | LoRA | QLoRA |
|-----------------|------|-------|
| Memory Usage    | Medium | Low (4-bit quantization) |
| Performance     | High | Nearly Full |
| VRAM Requirement | Moderate | Low (Consumer GPUs) |
| Fine-Tuning Speed | Fast | Faster |
| Parameter Updates | Low-rank adapters | Quantized + Low-rank adapters |

## 2. Preparing the Environment

### Install Dependencies
```bash
pip install torch transformers peft bitsandbytes datasets accelerate
```

### Set Up GPU & CUDA
Ensure CUDA and `bitsandbytes` support for **4-bit quantization**:
```bash
pip show bitsandbytes
```

## 3. Fine-Tuning Process

### Load Pretrained Model with QLoRA
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import torch

model_name = "deepseek-llm"
tokenizer = AutoTokenizer.from_pretrained(model_name)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto")
```

### Configure QLoRA Adapters
```python
config = LoraConfig(
    r=16, 
    lora_alpha=32, 
    target_modules=["q_proj", "v_proj"], 
    lora_dropout=0.05, 
    bias="none"
)
model = get_peft_model(model, config)
```

### Apply LoRA Fine-Tuning
```python
from peft import prepare_model_for_int8_training
model = prepare_model_for_int8_training(model)
```

### Prepare Dataset
```python
from datasets import load_dataset
dataset = load_dataset("your_dataset")
```

### Training with QLoRA
```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    save_steps=500,
    save_total_limit=2,
    logging_dir="./logs",
    evaluation_strategy="steps",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
)

trainer.train()
```

## 4. Saving & Loading the Fine-Tuned Model
```python
model.save_pretrained("./fine_tuned_deepseek")
tokenizer.save_pretrained("./fine_tuned_deepseek")
```

## 5. Deploying the Fine-Tuned Model
Use the fine-tuned model for inference:
```python
from transformers import pipeline
pipe = pipeline("text-generation", model="./fine_tuned_deepseek", tokenizer="./fine_tuned_deepseek")
print(pipe("Hello, how are you?"))
```

## Conclusion
Using LoRA and QLoRA, DeepSeek LLM can be fine-tuned efficiently on consumer GPUs, reducing hardware constraints while achieving high-quality outputs.


