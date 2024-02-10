# SyDNEy: Stable Diffusion Numerical Explorer

![SyDNEy_screenshot_20240206](https://github.com/pfeaster/SyDNEy/assets/10981679/77371473-5ac7-44fa-819f-45f9828cbafd)

SyDNEy is an interface for Stable Diffusion that offers a number of novel features.  It's written in Python, but it allows users to compose more or less elaborate programs for generating images and sounds in its own simplified scripting language, including various loops and batch processes.  Some things you can do with SyDNEy:
* Alternate among different models between steps.
* Manipulate prompt embeddings numerically, exploiting SD's own internal classification schemes (which you CAN learn to use!).
* Manipulate noise latents numerically -- or flip, concatenate, or invert them.
* Output stereo audio from spectrograms generated using Riffusion-like models.
* Create longer sound files made up of multiple generated audio clips joined cleanly together.
* Generate batches of images by iterating any parameter or parameters (seed, guidance scale, step count, etc.).

# Required Packages
For basic functionality, you’ll need diffusers, transformers, torch, torchvision, and Pillow (PIL).  If you want to take advantage of a discrete Nvidia GPU—which you’ll want to do if you can—be sure to install an appropriate version of torch with CUDA support.  Several other libraries are optional but necessary for some of SyDNEy’s specialized features: for audio output you’ll need numpy, soundfile, audio2numpy, torchaudio, and scipy, while for video output you’ll need OpenCV (cv2).

# Instructions
Create a folder for SyDNEy and place the SyDNEy.py file in it.  You can then run SyDNEy as you would any other Python script.  Within its folder, SyDNEy will create a couple subfolders the first time you run it, named SyDNEy_work (used for saving results) and SyDNEy_ref (used for configuration files, script backups, logs, and so forth).  For more details and a basic user guide including an introduction to SyDNEy's scripting language, see https://griffonagedotcom.wordpress.com/2024/02/07/introducing-sydney-stable-diffusion-numerical-explorer/
