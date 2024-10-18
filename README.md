# HoloLens-Streaming-CV-Example
Welcome to this repo :)) It contains a template project for conducting CV "on" the HoloLens 2. To achieve this, we have to stream sensor data from the HoloLens to an external PC (with GPU), run CV (and other ML models), and stream json data back to the HoloLens for processing and rendering. Yep... HoloLens can't really do on-device ML. Let's get into what is in this repo!

This package depends quite heavily on hl2ss: https://github.com/jdibenes/hl2ss, which depends on HoloLens2ForCV: https://github.com/microsoft/HoloLens2ForCV. Don't worry, the necessary files from these packages are already included in this repo. Just giving credit is all :))

## Unity Client
The Unity Client sits in the "unity" folder. The main scene is located in Assets/Scenes/Main.unity

In the main scene, there is a UI panel named "VASlate". It's to display the received string. It's for debugging purposes. Before deployment to participants, I recommend you disactivate this gameobject.

To build, you must be in Universal Windows Platform (UWP). Change build settings to UWP if this is not already the case. Then, set the architecture to ARM64. It will not run on any other architecture.

Once the Project finishes build, open it in Visual Studio 2022 (does not work in VS 2019 or VS Code). You must also have C++ 143 and Unity for Game Development packages installed through the visual studio installer. Connect your HoloLens 2 to the computer via USB-C.

To use this, you must also enable "Research Mode" within the HoloLens 2 (See this link for tutorial / more info - https://learn.microsoft.com/en-us/windows/mixed-reality/develop/advanced-concepts/research-mode).

- Open Start Menu > Settings and select Updates & Security.
- Select For Developers and enable Developer Mode.
- Scroll down and enable Device Portal.
- Under Device Portal, type in the url under "Wi-Fi". This will open the Device Portal on your computer.
- Go to System > Research Mode in the Device Portal.
- Select Allow access to sensor stream.
- Restart the HoloLens from the Power menu item at the top of the page or by physically pressing the power button on the HoloLens.

You now have Research Mode enabled!

Within the Unity code, the only important part of the code is within a script called IPCSkeleton.cs, which can be found in Assets/Scripts/IPCSkeleton.cs. Line 69 is "resultText.text = str;". This is where we set the content for the VASlate. "str" is the string that you sent to the HoloLens from the Python server. Do what you want to do with this string :))

## Python Server
The Python Server sits in the "python" folder. The main script is main.py

For the Python Server, start by creating a conda environment. I recommend Python 3.9, but anything above 3.8 should be fine.
```
conda create -n "hololens-streaming-cv" python=3.9
```

Then, install the following packages:

- OpenCV
```
pip install opencv-python
```
- PyAV
```
pip install av
```
- numPy
```
pip install numpy
```
- pynput
```
pip install pynput
```
- scipy
```
pip install scipy
```

This demo also relies on YOLOv8. So install ultranalytics
```
pip install ultralytics
```

Lastly, install Pytorch. I recommend using conda instead of pip. I also recommend uninstalling torch and torchvision just to check if some cpu version was installed.
```
conda remove torch torchvision
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

Okay! Now you can run this! First, edit line 23
```
host = '10.0.0.117'
```
This is in the HoloLens settings:
- Open Start Menu > Settings and select Updates & Security.
- Select For Developers.
- Scroll down to Device Portal.
- Under Device Portal, check the url under "Wi-Fi".
- Your host variable should be set to this url minus the https stuff. So if it reads "https://10.0.0.117", then just type 10.0.0.117

Then, run the HoloLens 2 app. Then, simply run in the terminal
```
python main.py
```

On the Python side, you should edit lines 169 - 186. This is where we handle YOLOv8 results. Edit this logic, store responses as a dictionary, and later convert it to a string (via str() method) before sending it to the Unity side.

Line 168 is
```
DISPLAY_MAP[port](port, data.payload)
```
This is for debugging purposes. It will create a window that shows you the received frames from the HoloLens 2. Feel free to comment this before deployment.

Line 26-40 has the ports variable. Commenting and uncommenting lines inside will enable the HoloLens to sent more data over to the Python server. You should uncomment: 
```
hl2ss.StreamPort.RM_VLC_LEFTFRONT
hl2ss.StreamPort.RM_VLC_RIGHTFRONT
```
These can be used for stereo depth estimation. 

If you want to do CV, you shoudl uncomment:
```
hl2ss.StreamPort.PERSONAL_VIDEO
```
This should be uncommented by default.

Enjoy! :)) Reach out to Jae via Slack if you have any questions.
