# PyConvolutionPerfusion

# Abstract  
This is a Perfusion blood flow analysis tool. It is mainly intended for MRI.  
It is classified as a deconvolution method, but uses "Convolution" instead of Deconvolution.  

<!-- 
In general, the Impulse Response is obtained by applying Deconvolution to the Arterial Input Function and the target Time Density Function.  
However, this tool sets an appropriate initial value for the Impulse response, convolves it with the Arterial Input Function, and optimizes the resulting composite curve and the Time Density Function of the target.  
-->

# Libraries  
Python 3.7  
torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113  
matplotlib==3.1.3  
opencv-python-headless==3.4.9.33  

# Impulse Response

![2inputs_1280x480_fps25](https://user-images.githubusercontent.com/106053283/176849366-cd24cfc8-75a3-4186-8cbf-592949e26ea5.gif)

![delay_1280x480_fps25](https://user-images.githubusercontent.com/106053283/176849473-070103ef-c229-494e-a1b4-bf473eeb4572.gif)

![missInput_1280x480_fps25](https://user-images.githubusercontent.com/106053283/176849483-81714919-14c7-4c54-93a7-0f095237565d.gif)


# Blood Flow
![Figure_1](https://user-images.githubusercontent.com/106053283/177734578-ccb45acb-2ed7-48a3-9e5b-2486a5b33c4e.png)
