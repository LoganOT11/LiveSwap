

## Dependencies
Beyond code dependencies, in its current form we rely on OBS virtual camera, and VB Audio

Install both applications. Set a new OBS scene with 
1. Window capture to catch input
2. Audio Input Capture to catch CABLE OUTPUT

In windows sound mixer (Win11) set the output of the window playing the content to CABLE INPUT

In main.py ensure that virtual cam and CABLE OUTPUT are selected for STREAM_DEVICE_INDEX and AUDIO_DEVICE_INDEX respectively

device_info.py can be used to find the appropriate indices for these devices.

## Running
py main.py
- For running input/output model
python portal/portal.py 
- For running web portal to drop content into