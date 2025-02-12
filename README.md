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

## User Interaction Flow

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
## Backup & Recovery Plan


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
## Full Process Overview

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



