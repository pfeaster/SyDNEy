# SyDNEy: Stable Diffusion Numerical Explorer

![SyDNEy_screenshot_20240206](https://github.com/pfeaster/SyDNEy/assets/10981679/77371473-5ac7-44fa-819f-45f9828cbafd)

SyDNEy is an interface for Stable Diffusion that offers a number of novel features.  It's written in Python, but it allows users to compose more or less elaborate programs for generating images and sounds in its own specialized scripting language, including various loops and batch processes.  Some things you can do with SyDNEy:
* Alternate among different models between steps.
* Manipulate prompt embeddings numerically, exploiting SD's own internal classification schemes (which you CAN learn to use!).
* Manipulate noise latents numerically -- or flip, concatenate, or invert them.
* Output stereo audio from spectrograms generated using Riffusion-like models.
* Create longer sound files made up of multiple generated audio clips joined cleanly together.
* Generate batches of images by iterating any parameter or parameters (seed, guidance scale, step count, etc.).

For more details and a basic user guide, see https://griffonagedotcom.wordpress.com/2024/02/07/introducing-sydney-stable-diffusion-numerical-explorer/
