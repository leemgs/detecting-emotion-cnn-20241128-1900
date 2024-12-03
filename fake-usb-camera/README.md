### Setting Up and Testing a Fake USB Camera with Python

#### **1. Install the Virtual Loopback Device**
To create a virtual USB camera device, you need to install and configure the `v4l2loopback` kernel module.

```bash
sudo apt-get install v4l2loopback-dkms
```

Create a configuration file to set up the virtual device with the desired options:

```bash
echo options v4l2loopback devices=1 video_nr=20 card_label="fakecam" exclusive_caps=1 | sudo tee -a /etc/modprobe.d/fakecam.conf
echo v4l2loopback | sudo tee -a /etc/modules-load.d/fakecam.conf
```

Unload and reload the kernel module to apply the changes:

```bash
sudo modprobe -r v4l2loopback
sudo modprobe v4l2loopback
```

Verify that the `v4l2loopback` module is loaded:

```bash
lsmod | grep v4l2
```

---

#### **2. Create and Run the Python Script for Fake Frames**

Create a Python script named `red_blue.py`:

```bash
vi ./emotion-red-blue.py
```

Add the following code to generate alternating red and blue frames:

```python
#!/usr/bin/env python3

import time
import pyfakewebcam
import numpy as np

blue = np.zeros((480, 640, 3), dtype=np.uint8)
blue[:, :, 2] = 255

red = np.zeros((480, 640, 3), dtype=np.uint8)
red[:, :, 0] = 255

camera = pyfakewebcam.FakeWebcam('/dev/video20', 640, 480)

while True:
    camera.schedule_frame(red)
    time.sleep(1/30.0)

    camera.schedule_frame(blue)
    time.sleep(1/30.0)
```

Save and exit the editor. Then run the script:

```bash
python3 ./red_blue.py
```

---

#### **3. Verify the Fake USB Camera**

Install the `ffmpeg` package:

```bash
sudo apt install ffmpeg
```

Check the output of the fake USB camera using `ffplay`:

```bash
ffplay /dev/video20
```

If the setup is successful, you should see alternating red and blue frames being displayed.
