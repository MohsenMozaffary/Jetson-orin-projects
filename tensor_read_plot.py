import cv2
import numpy as np
import threading
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from gdx import gdx
import argparse
from tensorly.decomposition import tucker

gdx_instance = gdx.gdx()

gdx_instance.open(connection='usb')
gdx_instance.select_sensors()
gdx_instance.start()

data = []

thermal_tensor = None
counter = 0

signal1 = np.array([])
signal2 = np.array([])

fs = 6.0

def update_data():
    global data
    while True:
        measurements = gdx_instance.read()
        if measurements is None:
            break
        data.append(measurements)

def update_plot(frame):
    plt.clf()

    plt.subplot(3, 1, 1)
    plt.plot(data, label='Raw Data')
    plt.xlabel('Time')
    plt.ylabel('Measurement')
    plt.title('Raw Data')

    plt.subplot(3, 1, 2)
    if signal1.size > 0:
        plt.plot(signal1, label='Signal 1 Filtered')
        plt.xlabel('Time')
        plt.ylabel('Measurement')
        plt.title('Signal 1')

    plt.subplot(3, 1, 3)
    if signal2.size > 0:
        plt.plot(signal2, label='Signal 2 Filtered')
        plt.xlabel('Time')
        plt.ylabel('Measurement')
        plt.title('Signal 2')

    plt.tight_layout()

def perform_tensor_decomposition():
    global thermal_tensor, counter, signal1, signal2
    while True:
        if counter // 5 == 59 and thermal_tensor is not None:
            tensor_to_decompose = thermal_tensor  # Take the latest 30 frames
            factors = tucker(tensor_to_decompose, rank=[2, 2, 2])
            core, factors = factors
            extracted_signal1 = factors[0][:, 0]  # Example: Extracting the first component of the first mode
            extracted_signal2 = factors[0][:, 1]  # Example: Extracting the second component of the first mode
            
            # Filter signals
            global signal1, signal2
            signal1 = butter_bandpass_filter(extracted_signal1, 0.1, 1, fs)
            signal2 = butter_bandpass_filter(extracted_signal2, 0.1, 1 , fs)
            
            print("Performing Tucker decomposition and filtering")

from scipy.signal import butter, filtfilt

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def capture_video():
    global thermal_tensor, counter
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0, help="Video device index")
    args = parser.parse_args()

    if args.device:
        dev = args.device
    else:
        dev = 2

    cap = cv2.VideoCapture(f'/dev/video{dev}', cv2.CAP_V4L)
    cap2 = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L)
    thermal_tensor = np.zeros((60, 192, 256))  
    counter = 0
    while True:
        ret, frame = cap2.read()
        ret2, frame2 = cap.read()

        if ret and ret2:
            cv2.imshow('RGB', frame)
            cv2.imshow('Thermal', frame2[:192,:,:])
            if counter < 300:
                if counter % 5 == 0:
                    thermal_tensor[counter//5] = frame2[:192,:,0]
            else:
                counter = 0
                thermal_tensor[0] = frame2[:192,:,0]
            counter += 1

        keyPress = cv2.waitKey(3)
        if keyPress == ord('q'):
            break

    cap.release()
    cap2.release()
    cv2.destroyAllWindows()

data_thread = threading.Thread(target=update_data)
video_thread = threading.Thread(target=capture_video)
tensor_thread = threading.Thread(target=perform_tensor_decomposition)

data_thread.start()
video_thread.start()
tensor_thread.start()

fig = plt.figure()
ani = animation.FuncAnimation(fig, update_plot, interval=33)

try:
    plt.show()
except KeyboardInterrupt:
    print("Plot interrupted.")

data_thread.join()
video_thread.join()
tensor_thread.join()

gdx_instance.stop()
gdx_instance.close()
