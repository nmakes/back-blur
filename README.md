# Back-Blur
## A simple demonstration of Background removal / blurring

### Requirements

Tested on python 3.6.9 with the following packages installed:

	opencv-python==3.4.2.17
	numpy==1.18.3

### Execution

Run the following command

	python3 camera.py

To use background removal (using a false background), set (Line 258)

	mode = 'remove'

For background blurring, set

	mode = 'blur'

For neither of them (raw camera output), set

	mode = None

### Sample

For best results, sit in a well lit area, with a plain background, and uncover your face.

![Demonstration](/images/demo_backblur.png)