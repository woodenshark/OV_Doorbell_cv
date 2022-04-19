# OV Test project

## Goal

Aim is to detect visitor's face and inform whether the person is known (family, facility member, friend, etc.) or a stranger. Target hardware is Raspberry Pi 4 (4Gb) and OV5647 camera module.

Goal is to implement:
 - Video/lmage capturing
 - Person and Face detection
 - Logic to match detected faces to pre-defined dataset of known faces

Output:
 - To simulate the voice informing of the known face - the program should print out the matching name or "Stranger" (Person with no face detected or non-matching face)
 - Collect and log to file total number of successful face detections
 - Collect and log to file total number of know personal detected
 - Collect and log to file total number of strangers detected
 - For each stranger detection - encode the image and save to file in separate folder "detected_strangers", with inner folder for each day with name format YYYYMMDD

## Installation

Install 32-bit desktop OS by the [Raspberry Pi Image Tool](https://www.raspberrypi.com/software/). After booting and setting things up increase GPU memory size (since 76 MBytes can be somewhat small for vision projects) using the following menu

<img src="https://github.com/woodenshark/OV_Doorbell_cv/raw/belezyakov_doc/images/raspi_config_1.jpg" width="400" height="380"> <img src="https://github.com/woodenshark/OV_Doorbell_cv/raw/belezyakov_doc/images/raspi_config_2.jpg" width="420" height="380">

With a fresh and clean Raspbian operating system, check the EEPROM software version. Check, and if needed update, the EEPROMs with the following commands:
```
# to get the current status
$ sudo rpi-eeprom-update
# if needed, to update the firmware
$ sudo rpi-eeprom-update -a
$ sudo reboot
```

The next step is to increase your swap space. OpenCV needs a lot of memory to compile from scratch. The latest versions want to see a minimum of 6.5 GB of memory before building. Your swap space is limited to 2048 MByte by default. To exceed this 2048 MByte limit, you will need to increase this maximum in the dphys-swapfile (both in /etc and /sbin folders). Using an editor of your choice change ```CONF_MAXSWAP``` variable in ```/sbin/dphys-swapfile``` to ```4096``` (default value is 2048) and ```CONF_SWAPSIZE``` to ```4096``` in ```/etc/dphys-swapfile``` (default value is 100). Call ```sudo rebooot``` after changes, it will need some more time for booting.

Before running installation script check memory first with ```free -m``` command - the sum should be more than 6.5 Gb. Then check mode of ```install.sh``` and change it if needed to 755 ```sudo chmod 755 ./install.sh```. Finally, run the script ```./install.sh``` and give it about 2 hours to finish.

After installation is completed, restore swapsize-variables in ```/etc/dphys-swapfile``` and ```/sbin/dphys-swapfile```.

Install required python packages by running:
```
pip3 install -r requirements.txt
```

For working with raspicam current release needs legacy camera support enabled. Call ```sudo raspi-config```, go to ```3 Interface Options``` and enable ```I1 Legacy Camera```. After reboot you are ready to go.

## Run
To run detector application call ```python3 -m app -v [video_source]``` providing video source - camera device index, video file path or "pi" for raspicam usage.

There are two tool scripts (in tools folder) using which you can update face model - call headshots.py script to add new person (or update existing one) photos, to retrain model with new dataset call train_model.py script. The model and dataset are used in unittest (app/test_face_recognizer.py) for recognizing all known persons of dataset by trained model.

## Limitations
 - First step of implemented algorithm searches person type objects in centered square area, since used MobileNet model trained on such images.
 - Current version excludes inner boxes of person type objects detected.
 - Since object tracker is being used, after attendace procedure, tracked id will be matched with attendant person name (overwriting face detected).

## Known issues
 - Software not working on 64-bit Raspberry OS.
 - Legacy camera support is needed.
 - Delay in main detection loop is mainly used for working with video files and depends on target platform and video file fps value.

## Links
 - https://qengineering.eu/install-opencv-4.5-on-raspberry-pi-4.html
 - https://github.com/raspberrypi/userland/issues/688
