# SyDNEy: Stable Diffusion Numerical Explorer

![SyDNEy_screenshot_20240206](https://github.com/pfeaster/SyDNEy/assets/10981679/77371473-5ac7-44fa-819f-45f9828cbafd)

SyDNEy is an interface for Stable Diffusion that offers a number of novel features.  It's written in Python, but it allows users to compose more or less elaborate programs for generating images and sounds in its own simplified scripting language, including various loops and batch processes.  Some things you can do with SyDNEy:
* Alternate among different models between steps.
* Manipulate prompt embeddings numerically, exploiting SD's own internal classification schemes (which you CAN learn to use!).
* Manipulate noise latents numerically -- or flip, concatenate, or invert them.
* Output stereo audio from spectrograms generated using Riffusion-like models.
* Create longer sound files made up of multiple generated audio clips joined cleanly together.
* Generate batches of images by iterating any parameter or parameters (seed, guidance scale, step count, etc.).

# Installation
1. Make sure you have [Python](https://www.python.org/downloads/) installed, version 3.10 or later.
2. Open a terminal (Command Prompt in Windows, Terminal in Mac/Linux).  You'll enter the commands that follow into it.
3.  Steps 3 and 4 are optional but recommended; they'll keep SyDNEy from interfering with any other Python packages you might want to use.  First, create a virtual environment:`python -m venv sydney-env`
4. Then activate it.  On Windows, enter `sydney-env\Scripts\activate`; on Mac/Linux, enter `source sydney-env/bin/activate`. 
5. If you have a discrete Nvidia GPU and want to take advantage of it when using SyDNEy, install a CUDA-enabled version of Torch: `pip install torch==2.12.0+cu122 torchvision==0.16.2+cu122 torchaudio==2.12.1+cu122 --extra-index-url https://download.pytorch.org/whl/cu122`
6. Install SyDNEy: `pip install git+https://github.com/pfeaster/SyDNEy.git`
7. Now just enter `sydney` and a SyDNEy window should open.

# Instructions
To launch SyDNEy, repeat steps 2, 3, 4, and 7 as described above under Installation -- or just steps 2 and 7 if you skipped steps 3 and 4 previously.

The first time you run SyDNEy, it will create a couple subfolders within its installation folder named SyDNEy_work (used for saving results) and SyDNEy_ref (used for configuration files, script backups, logs, and so forth).  Within the SyDNEy GUI, you can click "Load Backup," "Query Image," or "Query Script" to open a navigation window and see where these folders have been placed.  

For more details and a basic user guide including an introduction to SyDNEy's scripting language, see https://griffonagedotcom.wordpress.com/2024/02/07/introducing-sydney-stable-diffusion-numerical-explorer/
