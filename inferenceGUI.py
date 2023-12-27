import numpy as np
import tkinter as tk
from tkinter import ttk
import serial
import threading
import queue
import re


class inferenceGUI:
    def __init__(
            self, 
            X_test, 
            y_test, 
            acc_max,
            gyro_max,
            port='COM3', 
            baudrate=115200, 
            MCU = 'STM32', # 'STM32' or 'MAX'
            confidence = 0.95
            ):
        

        self.acc_max = acc_max
        self.gyro_max = gyro_max
        self.MCU = MCU
        self.port = port
        self.baudrate = baudrate
        self.X_test = X_test
        self.y_test = y_test

        if not (self.X_test.dtype == np.int8 or self.X_test.dtype == np.int16):
            self.X_test = self.X_test.astype(np.float32) # (1, 50, 6)
        if self.MCU == 'MAX' and self.X_test.dtype == np.float32:
            self.X_test[:,:,[0,1,2]] = ((self.X_test[:,:,[0,1,2]] + self.acc_max) / (self.acc_max*2) - 0.5) * 256
            self.X_test[:,:,[3,4,5]] = ((self.X_test[:,:,[3,4,5]] + self.gyro_max) / (self.gyro_max*2) - 0.5) * 256
        
        #self.y_pred = y_pred
        self.test_data = iter(zip(X_test, y_test))
        self.total = 0
        self.correct = 0
        self.confidence = confidence
        self.pred = 0
        self.queue = queue.Queue()
        self.mode = -1
        self.task = None
        self.serial_in_use = False
        
    
    def launch(self):
        
        self.pause_event = threading.Event()
        self.pause_event.set() 

        self.pause_mode_auto = threading.Event()
        self.pause_mode_auto.set()
        self.setup_UI()

    def setup_UI(self):

        self.root = tk.Tk()
        self.root.geometry("1024x768")
        self.root.title("Fall Detection")
        self.root.resizable(True, True)
        self.pred_cache = tk.StringVar()
        self.read_serial_data()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.mainloop()

    def on_close(self):
        # Close the serial connection
        if hasattr(self, 'ser') and self.ser.is_open:
            self.ser.close()
        # Destroy the window
        self.root.destroy()

    def read_serial_data(self):

        self.ser = serial.Serial(port=self.port, baudrate=self.baudrate, timeout=3)   
        # flush the serial port
        self.ser.flush()
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()
        self.welcome_page()
        self.pause_event = threading.Event()
        self.pause_event.set()  
        # Start a new thread to update the page
        threading.Thread(target=self.serial_daemon, daemon=True).start()

    def test_once_on_MCU(self):
        x, y = next(self.test_data)
        
        # shape of x needs to be (1, 50, 6), and type needs to be float32
        x = np.expand_dims(x, axis=0)
        
        x_bytes = x.tobytes()

        while self.serial_in_use:
            pass
        self.serial_in_use = True
        self.ser.reset_input_buffer()
        self.ser.write(b'Connect\r\n')
        line = self.ser.readline().decode('utf-8').strip()
        if line == 'Echo':
            self.ser.write(x_bytes)
            line = self.ser.readline().decode('utf-8').strip() # z.B.-127,127
            if line == 'Data received.':
                print('Data sent to MCU.')
            else:
                print('Data sending timeout.')
        else:
            print('Handshake timeout')
        self.serial_in_use = False
        (output1, output2) = self.queue.get(timeout=3)
        if output1 >= output2:
            y_pred = 0
        else:
            y_pred = 1
        self.total += 1
        if y == y_pred:
            self.correct += 1

        if y == 0:
            self.label_text.set('Not Falling')
            self.left_frame.config(bg='green')
            self.label.config(background='green')
            self.label2.config(background='green')
        else:
            self.label_text.set('Falling')
            self.left_frame.config(bg='red')
            self.label.config(background='red')
            self.label2.config(background='red')
        if y_pred == 0:
            self.prediction_text.set('Not Falling')
            self.right_frame.config(bg='green')
            self.prediction_label.config(background='green')
            self.prediction_label2.config(background='green')
        else:
            self.prediction_text.set('Falling')
            self.right_frame.config(bg='red')
            self.prediction_label.config(background='red')
            self.prediction_label2.config(background='red')
        self.task = None


    def reset(self):
        # Reset the page
        self.pred_label.config(text="")
        self.reset_button.destroy()
        self.pause_event.set() 
  

    def serial_daemon(self):

        while True:
       
            while self.serial_in_use:
                pass
            self.serial_in_use = True
            if self.ser.in_waiting:
                line = self.ser.readline().decode('utf-8').strip()
                print(line)
                if line.find('Inference completed') != -1:
                    match = re.search(r"output=\[(.*), (.*)\], elapsed time: (\d+)", line)
                    if match:
                        output1 = int(match.group(1))
                        output2 = int(match.group(2))
                        elapsed_time = int(match.group(3))
                        print(f'Inference output received, output1: {output1}, output2: {output2}, inference delay: {elapsed_time}us.')
                        if self.mode == 1:
                            self.queue.put((output1, output2))
                        elif self.mode == 2:
                            # self.pause_event.wait() 
                            if output1 >= output2 or output2 <= self.confidence*127:
                            # if output1>=output2:
                                self.pred_label.config(text="You are safe!")
                            else:
                                self.pred_label.config(text="You are Falling!!! Call 112!!!")
                                
              
                # mode 1: inference on PC
                elif line == 'Mode 1 selected.':
                    self.mode = 1
                    self.show_inference_from_PC_page()

                # mode 2: inference on MCU (only available for STM32)
                elif line == 'Mode 2 selected.' and self.MCU == 'STM32':

                    self.mode = 2
                    self.show_inference_from_MCU_page()

                # mode 0: idle
                elif line == 'Mode 0 selected.':

                    self.mode = 0
                    self.show_idle_page()
                    
                elif line.find('Inference failed') != -1:
                    print('Inference failed on MCU.')

            self.serial_in_use = False
            self.root.update()
            
    def clear_page(self):
        for widget in self.root.winfo_children():
            widget.destroy()
    
    def welcome_page(self):

        self.clear_page()

        self.welcome_label = ttk.Label(self.root, text="Please press the switch button \nto switch mode..", font=("Arial", 40))
        self.welcome_label.pack(expand=True)


    def show_inference_from_PC_page(self):
        # Clear the current content
        self.clear_page()

        # reset the self.task when going back to this page
        self.task = None

        # Central frame for mode selection
        self.mode_selection_frame = ttk.Frame(self.root)
        self.mode_selection_frame.pack(expand=True)

        # Label for mode selection
        self.mode_label = ttk.Label(self.mode_selection_frame, text="Select prediction mode:", font=("Arial", 40))
        self.mode_label.pack(pady=(10, 20))  

        # Frame to hold buttons
        self.button_frame = ttk.Frame(self.mode_selection_frame)
        self.button_frame.pack(pady=10)

        # Create a style
        style_button = ttk.Style()
        # Configure the style
        style_button.configure('TButton', font=('Arial', 25))  

        # Auto mode button
        self.auto_mode_button = ttk.Button(self.button_frame, text="Auto", command=self.enter_auto_mode, style='TButton')
        self.auto_mode_button.pack(side=tk.LEFT, padx=10)

        # Manual mode button
        self.manual_mode_button = ttk.Button(self.button_frame, text="Manual", command=self.enter_manual_mode, style='TButton')
        self.manual_mode_button.pack(side=tk.RIGHT, padx=10)
    

    def test_once_on_MCU_auto(self):
        
        for x, y in zip(self.X_test, self.y_test):
            self.pause_mode_auto.wait()
            if self.task is None:
                break

            # shape of x needs to be (1, 50, 6), and type needs to be float32
            x = np.expand_dims(x, axis=0)

            x_bytes = x.tobytes()

            while self.serial_in_use:
                pass
            self.serial_in_use = True
            self.ser.reset_input_buffer()
            self.ser.write(b'Connect\r\n')
            line = self.ser.readline().decode('utf-8').strip()
            if line == 'Echo':
                self.ser.write(x_bytes)
                line = self.ser.readline().decode('utf-8').strip() # z.B.-127,127
                print(line)
                if line == 'Data received.':
                    print('Data sent to MCU.')
                else:
                    print('Data sending timeout.')
            else:
                print('Handshake timeout')
            self.serial_in_use = False
            (output1, output2) = self.queue.get(timeout=3)
            if output1 >= output2:
                y_pred = 0
            else:
                y_pred = 1
            self.total += 1
            if y == y_pred:
                self.correct += 1
            self.accuracy = self.correct/self.total*100
            self.accuracy = round(self.accuracy, 2)
            # self.idx_label.config(text=f"Infering number {self.total} out of {self.num_total} set of data\n Accuracy: {self.accuracy}")
            self.idx_label.config(text=f"Infering number {self.total} out of {self.num_total} set of data")
            self.accuracy_label.config(text=f"Accuracy: {self.accuracy}%")
            
            print(f'Ground truth: {y}, prediction: {y_pred}')
            

    def enter_auto_mode(self):
        # Clear the current content
        self.clear_page()

        # reset the correct and total
        self.correct = 0
        self.total = 0
        self.current_idx = 0
        self.accuracy = 0
        self.num_total = len(self.y_test)
        # reinitialize the queue
        self.queue = queue.Queue()
        # TODO: Create the content for the self.correct, self.total and self.current_idx. They are all put inthe middle of the page
        
        # Create labels for the self.correct, self.total and self.current_idx
        # self.idx_label = ttk.Label(self.root, text=f"Infering number {self.total} out of {self.num_total} set of data\n Accuracy: {self.accuracy}", font=("Arial", 40, 'bold'))
        self.idx_label = ttk.Label(self.root, text=f"Infering number {self.total} out of {self.num_total} set of data", font=("Arial", 40, 'bold'))
        self.idx_label.grid(row=0, column=1)
        self.idx_label.place(relx=0.5, rely=0.5, anchor='center')
        # Create a label for the accuracy
        self.accuracy_label = ttk.Label(self.root, text=f"Accuracy: {self.accuracy}%", font=("Arial", 40, 'bold'))
        self.accuracy_label.place(relx=0.5, rely=0.6, anchor='center')

        # crate a frame 
        if self.task == None:
            self.task = threading.Thread(target=self.test_once_on_MCU_auto)
            self.task.start()
            

        # Back button
        self.back_button = ttk.Button(self.root, text="Back", command=self.show_inference_from_PC_page, style='TButton')
        self.back_button.pack(pady=10)
        self.back_button.place(relx=0.3, rely=1, anchor='s')

        # Pause button
        self.pause_button = ttk.Button(self.root, text="Pause", command=self.pause, style='TButton')
        self.pause_button.pack(pady=10)
        self.pause_button.place(relx=0.7, rely=1, anchor='s')

    
    def pause(self):
        if self.pause_mode_auto.is_set():
            self.pause_button.config(text="Resume")
            self.pause_mode_auto.clear()  # Pause the loop
        else:
            self.pause_button.config(text="Pause")
            self.pause_mode_auto.set() 

    def enter_manual_mode(self):
        self.clear_page()

        self.top_frame = tk.Frame(self.root)
        self.top_frame.pack(side='top', fill='both', expand=True)
        # Create a frame for the left half of the window
        self.left_frame = tk.Frame(self.top_frame, bd=2, relief='solid', bg='white')
        self.left_frame.pack(side='left', fill='both', expand=True)
        self.left_frame.pack_propagate(False)  # Prevent the frame from changing size

        # Create a frame for the right half of the window
        self.right_frame = tk.Frame(self.top_frame, bd=2, relief='solid', bg='white')
        self.right_frame.pack(side='left', fill='both', expand=True)
        self.right_frame.pack_propagate(False)  # Prevent the frame from changing size

        # Back button
        self.back_button = ttk.Button(self.root, text="Back", command=self.show_inference_from_PC_page, style='TButton')
        self.back_button.pack(side=tk.LEFT, padx=120, pady=10)

        # Next button
        self.next_button = ttk.Button(self.root, text="Next Prediction", command=self.on_button_click, style='TButton')
        self.next_button.pack(side=tk.RIGHT, padx=120, pady=10)

  
        self.label = tk.Label(self.left_frame, text='Ground Truth Label:', font=("Helvetica", 24), bg='white')
        self.label.pack()

        self.label_text = tk.StringVar()
        self.label2 = tk.Label(self.left_frame, textvariable=self.label_text, font=("Helvetica", 50), bg='white')
        self.label2.place(relx=0.5, rely=0.5, anchor='center')

        self.prediction_label = tk.Label(self.right_frame, text='Prediction:', font=("Helvetica", 24), bg='white')
        self.prediction_label.pack()

        # Create a label to display the prediction
        self.prediction_text = tk.StringVar()
        self.prediction_label2 = tk.Label(self.right_frame, textvariable=self.prediction_text, font=("Helvetica", 50), bg='white')
        self.prediction_label2.place(relx=0.5, rely=0.5, anchor='center')


    def on_button_click(self):
        # print('on_button_click')
        #y_pred = self.ser.readline().decode('utf-8').strip().split(',')# z.B.-127,127

        # take the highest value
        #y_pred = max(map(int, y_pred))
        if self.task == None:
            self.task = threading.Thread(target=self.test_once_on_MCU)
            self.task.start()
        

    def show_inference_from_MCU_page(self):
        # Clear the current content
        self.clear_page()

        # Create the label with the updated text
        self.pred_label = ttk.Label(self.root, font=("Arial", 50, 'bold'))
        self.pred_label.place(relx=0.5, rely=0.5, anchor='center')

        # Add a reset button
        # self.reset_button = ttk.Button(self.root, text="Reset", command=self.show_inference_from_MCU_page)
        # self.reset_button.place(relx=0.5, rely=0.6, anchor='center')
   

    def show_idle_page(self):
        # Clear the current content
        self.clear_page()

        # Create new content for idle page
        self.idle_label = ttk.Label(self.root, text="You are in Sleep Mode\nZZZ....\n\nPress the switch button \nto switch mode.", font=("Arial", 40))
        self.idle_label.pack(expand=True)
