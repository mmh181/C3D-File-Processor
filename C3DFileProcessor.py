import tkinter as tk
import os
import matplotlib.pyplot as plt
from tkinter import filedialog
import tkinter.messagebox
from ezc3d import c3d
import pandas as pd
from scipy import signal
from scipy.signal import butter, filtfilt, cheby1, cheby2, ellip, bessel
import pywt


def select_files():
    file_paths = filedialog.askopenfilenames(filetypes=[("C3D Files", "*.c3d")])
    if file_paths:
        for path in file_paths:
            listbox.insert(tk.END, path)

def select_output_dir():
    dir_path = filedialog.askdirectory()
    if dir_path:
        output_dir.set(dir_path)

def plot_data(file_path):
    # Read the data from the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)

    # Create a new figure for the plot
    fig = plt.figure()

    # Set the title of the plot to the file name
    plt.title(os.path.basename(file_path))

    # Create a plot of the data using matplotlib
    plt.plot(df)

    # Add axis labels to the plot
    #plt.xlabel("Time")
    #plt.ylabel("Position")

    # Add a legend to the plot
    plt.legend(df.columns, bbox_to_anchor=(1.0, 1.10), loc='upper left')

    # Show the plot in a non-blocking manner
    plt.show(block=False)

def select_csv_files():
    file_paths = filedialog.askopenfilenames(filetypes=[("CSV Files", "*.csv")])
    if file_paths:
        for file_path in file_paths:
            plot_data(file_path)

def update_wavelet_family_options(*args):
    #families = pywt.families()

    # Define the desired wavelet families
    desired_families = ["haar", "db", "sym", "coif", "bior", "rbio", "dmey"]
    
    # Get the available wavelet families from pywt that are also in the desired families list
    families = [family for family in pywt.families() if family in desired_families]
    
    menu = wavelet_family_optionmenu["menu"]
    menu.delete(0, "end")
    
    for family in families:
        menu.add_command(label=family, command=lambda value=family: wavelet_family_var.set(value))
    
    if families:
        wavelet_family_var.set(families[0])
    else:
        wavelet_family_var.set("")

def update_wavelet_options(*args):
    family = wavelet_family_var.get()
    if family:
        wavelets = pywt.wavelist(family)
    else:
        wavelets = []
    
    menu = wavelet_optionmenu["menu"]
    menu.delete(0, "end")
    
    for wavelet in wavelets:
        menu.add_command(label=wavelet, command=lambda value=wavelet: wavelet_var.set(value))
    
    if wavelets:
        wavelet_var.set(wavelets[0])
    else:
        wavelet_var.set("")

def convert_files():
    # Validate filter parameters
    filter_design = filter_design_var.get()
    
    # Check if any C3D files have been selected
    c3d_files = listbox.get(0, tk.END)
    if not c3d_files:
        tkinter.messagebox.showerror("Error", "Please select at least one C3D file.")
        return

    # Check if an output directory has been specified
    output_directory = output_dir.get()
    if not output_directory:
        tkinter.messagebox.showerror("Error", "Please specify an output directory.")
        return
    
    # Get common filter parameters
    try:
        if filter_design != "None" and filter_design != "Wavelet Transform (Discrete)":
            order_str = order_entry.get()
            if not order_str:
                raise ValueError("Please enter a value for the order of the filter.")
            order = int(order_str)
            if order <= 0:
                raise ValueError("Order must be a positive integer.")
            
            fs_str = fs_entry.get()
            fs = float(fs_str)
            if fs <= 0:
                raise ValueError("Sampling frequency must be a positive number.")
            nyq = 0.5 * fs

            filter_type = filter_type_var.get()

            if filter_type in ["Low Pass", "Band Pass"]:
                low_cutoff_str = low_cutoff_entry.get()
                low_cutoff = float(low_cutoff_str)
                if low_cutoff <= 0 or low_cutoff >= nyq:
                    raise ValueError(f"Low cutoff frequency must be between 0 and {nyq} Hz.")

            if filter_type in ["High Pass", "Band Pass"]:
                high_cutoff_str = high_cutoff_entry.get()
                high_cutoff = float(high_cutoff_str)
                if high_cutoff <= 0 or high_cutoff >= nyq:
                    raise ValueError(f"High cutoff frequency must be between 0 and {nyq} Hz.")
    except ValueError as e:
        tk.messagebox.showerror("Error", str(e))
        return

    if filter_design == "Butterworth":
        try:
            if filter_type == "Low Pass":
                wp = low_cutoff / nyq
                b, a = butter(order, wp, btype='low', analog=False)
                
            elif filter_type == "High Pass":
                wp = high_cutoff / nyq
                b, a = butter(order, wp, btype='high', analog=False)
                
            elif filter_type == "Band Pass":
                wp = [low_cutoff / nyq, high_cutoff / nyq]
                b, a = butter(order, wp, btype='band', analog=False)
        except ValueError as e:
            tk.messagebox.showerror("Error", str(e))
            return

    elif filter_design == "Chebyshev Type I":
        try:
            ripple_str = ripple_entry.get()
            ripple = float(ripple_str)
            if ripple <= 0:
                raise ValueError("Passband ripple must be a positive number.")
            
            if filter_type == "Low Pass":
                wp = low_cutoff / nyq
                b, a = cheby1(order, ripple, wp, btype='low', analog=False)
                
            elif filter_type == "High Pass":
                wp = high_cutoff / nyq
                b, a = cheby1(order, ripple, wp, btype='high', analog=False)
                
            elif filter_type == "Band Pass":
                wp = [low_cutoff / nyq, high_cutoff / nyq]
                b, a = cheby1(order, ripple, wp, btype='band', analog=False)
        except ValueError as e:
            tk.messagebox.showerror("Error", str(e))
            return

    elif filter_design == "Chebyshev Type II":
        try:
            attenuation_str = attenuation_entry.get()
            attenuation = float(attenuation_str)
            if attenuation <= 0:
                raise ValueError("Stopband attenuation must be a positive number.")
            
            if filter_type == "Low Pass":
                ws = low_cutoff / nyq
                b, a = cheby2(order, attenuation, ws, btype='low', analog=False)
                
            elif filter_type == "High Pass":
                ws = high_cutoff / nyq
                b, a = cheby2(order, attenuation, ws, btype='high', analog=False)
                
            elif filter_type == "Band Pass":
                ws = [low_cutoff / nyq, high_cutoff / nyq]
                b, a = cheby2(order, attenuation, ws, btype='band', analog=False)
        except ValueError as e:
            tk.messagebox.showerror("Error", str(e))
            return

    elif filter_design == "Elliptic (Cauer)":
        try:
            ripple_str = ripple_entry.get()
            ripple = float(ripple_str)
            if ripple <= 0:
                raise ValueError("Passband ripple must be a positive number.")
            
            attenuation_str = attenuation_entry.get()
            attenuation = float(attenuation_str)
            if attenuation <= 0:
                raise ValueError("Stopband attenuation must be a positive number.")
            
            if filter_type == "Low Pass":
                wp = low_cutoff / nyq
                b, a = ellip(order, ripple, attenuation, wp, btype='low', analog=False)
                
            elif filter_type == "High Pass":
                wp = high_cutoff / nyq
                b, a = ellip(order, ripple, attenuation, wp, btype='high', analog=False)
                
            elif filter_type == "Band Pass":
                wp = [low_cutoff / nyq, high_cutoff / nyq]
                b, a = ellip(order, ripple, attenuation, wp, btype='band', analog=False)
        except ValueError as e:
            tk.messagebox.showerror("Error", str(e))
            return

    elif filter_design == "Bessel":
        try:
            if filter_type == "Low Pass":
                wp = low_cutoff / nyq
                b, a = bessel(order, wp, btype='low', analog=False)
                
            elif filter_type == "High Pass":
                wp = high_cutoff / nyq
                b, a = bessel(order, wp, btype='high', analog=False)
                
            elif filter_type == "Band Pass":
                wp = [low_cutoff / nyq, high_cutoff / nyq]
                b, a = bessel(order, wp, btype='band', analog=False)
        except ValueError as e:
            tk.messagebox.showerror("Error", str(e))
            return

    elif filter_design == "Wavelet Transform (Discrete)":
        try:
            wavelet_family = wavelet_family_var.get()
            wavelet = wavelet_var.get()
            level_str = level_entry.get()
            
            if not level_str:
                raise ValueError("Please enter a value for the level of decomposition.")
            level = int(level_str)
            if level < 1:
                raise ValueError("Level of decomposition must be a positive integer.")
            
            level_recon_str = level_recon_entry.get()
            if not level_recon_str:
                raise ValueError("Please enter a value for the level of reconstruction.")
            level_recon = int(level_recon_str)
            if level_recon < 1:
                raise ValueError("Level of reconstruction must be a positive integer.")
            
            threshold_method = threshold_method_var.get()
            if not threshold_method:
                raise ValueError("Please select a thresholding method.")
            threshold_value_str = threshold_value_entry.get()
            if not threshold_value_str:
                raise ValueError("Please enter a value for the threshold value.")
            threshold_value = float(threshold_value_str)
            if threshold_value < 0:
                raise ValueError("Threshold value must be a non-negative number.")
            
        except ValueError as e:
            tk.messagebox.showerror("Error", str(e))
            return

    else: 
        hide_wavelet_options()
        hide_butterworth_options()

    for c3d_file in listbox.get(0, tk.END):
        c3d_data = c3d(c3d_file)
        point_labels = [label.replace(' ', '_') for label in c3d_data['parameters']['POINT']['LABELS']['value']]
        column_labels = []
        for label in point_labels:
            column_labels.extend([f"{label}_X", f"{label}_Y", f"{label}_Z"])
        data = c3d_data['data']['points'][0:3].transpose([2, 1, 0]).reshape(-1, c3d_data['header']['points']['size'] * 3)

        # Apply filter
        suffix = ""
        
        if filter_design == "Butterworth":
            
            suffix += f"_{filter_design.lower().replace(' ', '_')}"
            
            suffix += f"_{filter_type.lower().replace(' ', '_')}"
            
            if filter_type == "Low Pass":
                wp = low_cutoff / nyq
                b, a = butter(order, wp, btype='low', analog=False)
                suffix += f"_low_cutoff_{int(low_cutoff)}"
                
            elif filter_type == "High Pass":
                wp = high_cutoff / nyq
                b, a = butter(order, wp, btype='high', analog=False)
                suffix += f"_high_cutoff_{int(high_cutoff)}"
                
            elif filter_type == "Band Pass":
                wp = [low_cutoff / nyq, high_cutoff / nyq]
                b, a = butter(order, wp, btype='band', analog=False)
                suffix += f"_low_cutoff_{int(low_cutoff)}_high_cutoff_{int(high_cutoff)}"
                
            data = filtfilt(b, a, data)
            
        elif filter_design == "Chebyshev Type I":
            suffix += f"_{filter_design.lower().replace(' ', '_')}"
            suffix += f"_{filter_type.lower().replace(' ', '_')}"
            ripple_str = ripple_entry.get()
            ripple = float(ripple_str)
            suffix += f"_ripple_{ripple}"
            if filter_type == "Low Pass":
                wp = low_cutoff / nyq
                b, a = cheby1(order, ripple, wp, btype='low', analog=False)
                suffix += f"_low_cutoff_{int(low_cutoff)}"
            elif filter_type == "High Pass":
                wp = high_cutoff / nyq
                b, a = cheby1(order, ripple, wp, btype='high', analog=False)
                suffix += f"_high_cutoff_{int(high_cutoff)}"
            elif filter_type == "Band Pass":
                wp = [low_cutoff / nyq, high_cutoff / nyq]
                b, a = cheby1(order, ripple, wp, btype='band', analog=False)
                suffix += f"_low_cutoff_{int(low_cutoff)}_high_cutoff_{int(high_cutoff)}"
            data = filtfilt(b, a, data)

        elif filter_design == "Chebyshev Type II":
            suffix += f"_{filter_design.lower().replace(' ', '_')}"
            suffix += f"_{filter_type.lower().replace(' ', '_')}"
            attenuation_str = attenuation_entry.get()
            attenuation = float(attenuation_str)
            suffix += f"_attenuation_{attenuation}"
            if filter_type == "Low Pass":
                ws = low_cutoff / nyq
                b, a = cheby2(order, attenuation, ws, btype='low', analog=False)
                suffix += f"_low_cutoff_{int(low_cutoff)}"
            elif filter_type == "High Pass":
                ws = high_cutoff / nyq
                b, a = cheby2(order, attenuation, ws, btype='high', analog=False)
                suffix += f"_high_cutoff_{int(high_cutoff)}"
            elif filter_type == "Band Pass":
                ws = [low_cutoff / nyq, high_cutoff / nyq]
                b, a = cheby2(order, attenuation, ws, btype='band', analog=False)
                suffix += f"_low_cutoff_{int(low_cutoff)}_high_cutoff_{int(high_cutoff)}"
            data = filtfilt(b, a, data)

        elif filter_design == "Elliptic (Cauer)":
            suffix += f"_{filter_design.lower().replace(' ', '_')}"
            suffix += f"_{filter_type.lower().replace(' ', '_')}"
            ripple_str = ripple_entry.get()
            ripple = float(ripple_str)
            suffix += f"_ripple_{ripple}"
            attenuation_str = attenuation_entry.get()
            attenuation = float(attenuation_str)
            suffix += f"_attenuation_{attenuation}"
            if filter_type == "Low Pass":
                wp = low_cutoff / nyq
                b, a = ellip(order, ripple, attenuation, wp, btype='low', analog=False)
                suffix += f"_low_cutoff_{int(low_cutoff)}"
            elif filter_type == "High Pass":
                wp = high_cutoff / nyq
                b, a = ellip(order, ripple, attenuation, wp, btype='high', analog=False)
                suffix += f"_high_cutoff_{int(high_cutoff)}"
            elif filter_type == "Band Pass":
                wp = [low_cutoff / nyq, high_cutoff / nyq]
                b, a = ellip(order, ripple, attenuation, wp, btype='band', analog=False)
                suffix += f"_low_cutoff_{int(low_cutoff)}_high_cutoff_{int(high_cutoff)}"
            data = filtfilt(b, a, data)
            
        elif filter_design == "Bessel":
            
            suffix += f"_{filter_design.lower().replace(' ', '_')}"
            
            suffix += f"_{filter_type.lower().replace(' ', '_')}"
            
            if filter_type == "Low Pass":
                wp = low_cutoff / nyq
                b, a = bessel(order, wp, btype='low', analog=False)
                suffix += f"_low_cutoff_{int(low_cutoff)}"
                
            elif filter_type == "High Pass":
                wp = high_cutoff / nyq
                b, a = bessel(order, wp, btype='high', analog=False)
                suffix += f"_high_cutoff_{int(high_cutoff)}"
                
            elif filter_type == "Band Pass":
                wp = [low_cutoff / nyq, high_cutoff / nyq]
                b, a = bessel(order, wp, btype='band', analog=False)
                suffix += f"_low_cutoff_{int(low_cutoff)}_high_cutoff_{int(high_cutoff)}"
                
            data = filtfilt(b, a, data)
            
        elif filter_design == "Wavelet Transform (Discrete)":
            wavelet_family = wavelet_family_var.get()
            wavelet = wavelet_var.get()
            level_str = level_entry.get()
            level = int(level_str)
            coeffs = pywt.wavedec(data, wavelet, level=level)
            
             # Apply thresholding to the wavelet coefficients
            threshold_method = threshold_method_var.get()
            if threshold_method == "Hard":
                coeffs = [pywt.threshold(c, threshold_value) for c in coeffs]
            elif threshold_method == "Soft":
                coeffs = [pywt.threshold(c, threshold_value, 'soft') for c in coeffs]
            
            data = pywt.waverec(coeffs, wavelet)
            
            suffix += f"_{filter_design.lower().replace(' ', '_')}"
            suffix += f"_wavelet_family_{wavelet_family}_wavelet_type_{wavelet}_level_of_decomposition_{level}_level_of_reconstruction_{level_recon}_thresholding_method_{threshold_method}_threshold_value_{threshold_value}"

        else:
            suffix += "_no_filter"

        df = pd.DataFrame(data, columns=column_labels)
        output_file_name = f"{output_dir.get()}/{c3d_file.split('/')[-1].split('.')[0]}{suffix}.csv"
        df.to_csv(output_file_name, index=False)

def hide_wavelet_options():
    wavelet_family_label.grid_remove()
    wavelet_family_optionmenu.grid_remove()
    wavelet_label.grid_remove()
    wavelet_optionmenu.grid_remove()
    level_label.grid_remove()
    level_entry.grid_remove()
    level_recon_label.grid_remove()
    level_recon_entry.grid_remove()
    threshold_method_label.grid_remove()
    threshold_method_optionmenu.grid_remove()
    threshold_value_label.grid_remove()
    threshold_value_entry.grid_remove()

def show_wavelet_options():
    wavelet_family_label.grid()
    wavelet_family_optionmenu.grid()
    wavelet_label.grid()
    wavelet_optionmenu.grid()
    level_label.grid()
    level_entry.grid()
    level_recon_label.grid()
    level_recon_entry.grid()
    threshold_method_label.grid()
    threshold_method_optionmenu.grid()
    threshold_value_label.grid()
    threshold_value_entry.grid()

def hide_butterworth_options():
    order_label.grid_remove()
    order_entry.grid_remove()
    fs_label.grid_remove()
    fs_entry.grid_remove()
    filter_type_label.grid_remove()
    low_pass_rb.grid_remove()
    high_pass_rb.grid_remove()
    band_pass_rb.grid_remove()
    low_cutoff_label.grid_remove()
    low_cutoff_entry.grid_remove()
    high_cutoff_label.grid_remove()
    high_cutoff_entry.grid_remove()

def show_butterworth_options():
    order_label.grid()
    order_entry.grid()
    fs_label.grid()
    fs_entry.grid()
    filter_type_label.grid()
    low_pass_rb.grid()
    high_pass_rb.grid()
    band_pass_rb.grid()

def hide_cheby1_options():
    order_label.grid_remove()
    order_entry.grid_remove()
    fs_label.grid_remove()
    fs_entry.grid_remove()
    filter_type_label.grid_remove()
    low_pass_rb.grid_remove()
    high_pass_rb.grid_remove()
    band_pass_rb.grid_remove()
    low_cutoff_label.grid_remove()
    low_cutoff_entry.grid_remove()
    high_cutoff_label.grid_remove()
    high_cutoff_entry.grid_remove()
    ripple_label.grid_remove()
    ripple_entry.grid_remove()

def show_cheby1_options():
    order_label.grid()
    order_entry.grid()
    fs_label.grid()
    fs_entry.grid()
    filter_type_label.grid()
    low_pass_rb.grid()
    high_pass_rb.grid()
    band_pass_rb.grid()
    ripple_label.grid()
    ripple_entry.grid()

def hide_cheby2_options():
    order_label.grid_remove()
    order_entry.grid_remove()
    fs_label.grid_remove()
    fs_entry.grid_remove()
    filter_type_label.grid_remove()
    low_pass_rb.grid_remove()
    high_pass_rb.grid_remove()
    band_pass_rb.grid_remove()
    low_cutoff_label.grid_remove()
    low_cutoff_entry.grid_remove()
    high_cutoff_label.grid_remove()
    high_cutoff_entry.grid_remove()
    attenuation_label.grid_remove()
    attenuation_entry.grid_remove()

def show_cheby2_options():
    order_label.grid()
    order_entry.grid()
    fs_label.grid()
    fs_entry.grid()
    filter_type_label.grid()
    low_pass_rb.grid()
    high_pass_rb.grid()
    band_pass_rb.grid()
    attenuation_label.grid()
    attenuation_entry.grid()

def hide_ellip_options():
    order_label.grid_remove()
    order_entry.grid_remove()
    fs_label.grid_remove()
    fs_entry.grid_remove()
    filter_type_label.grid_remove()
    low_pass_rb.grid_remove()
    high_pass_rb.grid_remove()
    band_pass_rb.grid_remove()
    low_cutoff_label.grid_remove()
    low_cutoff_entry.grid_remove()
    high_cutoff_label.grid_remove()
    high_cutoff_entry.grid_remove()
    ripple_label.grid_remove()
    ripple_entry.grid_remove()
    attenuation_label.grid_remove()
    attenuation_entry.grid_remove()

def show_ellip_options():
    order_label.grid()
    order_entry.grid()
    fs_label.grid()
    fs_entry.grid()
    filter_type_label.grid()
    low_pass_rb.grid()
    high_pass_rb.grid()
    band_pass_rb.grid()
    ripple_label.grid()
    ripple_entry.grid()
    attenuation_label.grid()
    attenuation_entry.grid()

def hide_bessel_options():
    order_label.grid_remove()
    order_entry.grid_remove()
    fs_label.grid_remove()
    fs_entry.grid_remove()
    filter_type_label.grid_remove()
    low_pass_rb.grid_remove()
    high_pass_rb.grid_remove()
    band_pass_rb.grid_remove()
    low_cutoff_label.grid_remove()
    low_cutoff_entry.grid_remove()
    high_cutoff_label.grid_remove()
    high_cutoff_entry.grid_remove()

def show_bessel_options():
    order_label.grid()
    order_entry.grid()
    fs_label.grid()
    fs_entry.grid()
    filter_type_label.grid()
    low_pass_rb.grid()
    high_pass_rb.grid()
    band_pass_rb.grid()

def hide_high_cutoff_frequency():
    high_cutoff_label.grid_remove()
    high_cutoff_entry.grid_remove()

def show_high_cutoff_frequency():
    high_cutoff_label.grid()
    high_cutoff_entry.grid()

def hide_low_cutoff_frequency():
    low_cutoff_label.grid_remove()
    low_cutoff_entry.grid_remove()

def show_low_cutoff_frequency():
    low_cutoff_label.grid()
    low_cutoff_entry.grid()

def on_filter_design_change(*args):
    filter_design = filter_design_var.get()

    if filter_design == "Butterworth":
        show_butterworth_options()
        on_filter_type_change()

        ripple_label.grid_remove()
        ripple_entry.grid_remove()
        
        attenuation_label.grid_remove()
        attenuation_entry.grid_remove()

    elif filter_design == "Chebyshev Type I":
        show_cheby1_options()
        on_filter_type_change()

        attenuation_label.grid_remove()
        attenuation_entry.grid_remove()

    elif filter_design == "Chebyshev Type II":
        show_cheby2_options()
        on_filter_type_change()

        ripple_label.grid_remove()
        ripple_entry.grid_remove()

    elif filter_design == "Elliptic (Cauer)":
        show_ellip_options()
        on_filter_type_change()

    elif filter_design == "Bessel":
        show_bessel_options()
        on_filter_type_change()

        ripple_label.grid_remove()
        ripple_entry.grid_remove()
        
        attenuation_label.grid_remove()
        attenuation_entry.grid_remove()

    else:
        hide_butterworth_options()
        hide_cheby1_options()
        hide_cheby2_options()
        hide_ellip_options()
        hide_bessel_options()

        ripple_label.grid_remove()
        ripple_entry.grid_remove()
        
        attenuation_label.grid_remove()
        attenuation_entry.grid_remove()

    if filter_design == "Wavelet Transform (Discrete)":
        show_wavelet_options()
    else:
        hide_wavelet_options()
    


def on_filter_type_change(*args):
    filter_type = filter_type_var.get()
    
    if filter_type == "Low Pass":
        show_low_cutoff_frequency()
        hide_high_cutoff_frequency()
        
    elif filter_type == "High Pass":
        hide_low_cutoff_frequency()
        show_high_cutoff_frequency()
        
    elif filter_type == "Band Pass":
        show_low_cutoff_frequency()
        show_high_cutoff_frequency()
        
    
root = tk.Tk()
root.title("C3D File Processor")

welcome_label = tk.Label(root, text="Welcome to the C3D File Processor!")
welcome_label.pack()

select_files_button = tk.Button(root, text="Select C3D Files", command=select_files)
select_files_button.pack()

listbox = tk.Listbox(root)
listbox.pack()

output_dir_label = tk.Label(root, text="Output Directory:")
output_dir_label.pack()

output_dir = tk.StringVar()
output_entry = tk.Entry(root, textvariable=output_dir)
output_entry.pack()

select_output_button = tk.Button(root, text="Select Output Directory", command=select_output_dir)
select_output_button.pack()

filter_frame = tk.LabelFrame(root, text="Filter Options")
filter_frame.pack()

filter_design_label = tk.Label(filter_frame, text="Filter Design:")
filter_design_label.grid(row=0,column=0)

filter_design_var=tk.StringVar(value="None")
filter_design_var.trace("w", on_filter_design_change)
filter_design_optionmenu=tk.OptionMenu(filter_frame,
                                     filter_design_var,
                                     "None",
                                     "Butterworth",
                                     "Chebyshev Type I",
                                     "Chebyshev Type II",
                                     "Elliptic (Cauer)",
                                     "Bessel",
                                     "Wavelet Transform (Discrete)")
filter_design_optionmenu.grid(row=0,column=1)

order_label=tk.Label(filter_frame,text="Order:")
order_label.grid(row=1,column=0)

order_entry=tk.Entry(filter_frame,width=5)
order_entry.grid(row=1,column=1)

fs_label=tk.Label(filter_frame,text="Sampling Frequency (Hz):")
fs_label.grid(row=2,column=0)

fs_entry=tk.Entry(filter_frame,width=10)
fs_entry.grid(row=2,column=1)

filter_type_label=tk.Label(filter_frame,text="Filter Type:")
filter_type_label.grid(row=3,column=0)

low_cutoff_label=tk.Label(filter_frame,text="Low Cutoff Frequency (Hz):")
low_cutoff_label.grid(row=4,column=0)

low_cutoff_entry=tk.Entry(filter_frame,width=10)
low_cutoff_entry.grid(row=4,column=1)

high_cutoff_label=tk.Label(filter_frame,text="High Cutoff Frequency (Hz):")
high_cutoff_label.grid(row=5,column=0)

high_cutoff_entry=tk.Entry(filter_frame,width=10)
high_cutoff_entry.grid(row=5,column=1)

ripple_label = tk.Label(filter_frame, text="Passband Ripple (dB):")
ripple_label.grid(row=6, column=0)

ripple_entry = tk.Entry(filter_frame, width=10)
ripple_entry.grid(row=6, column=1)

attenuation_label = tk.Label(filter_frame, text="Stopband Attenuation (dB):")
attenuation_label.grid(row=7, column=0)

attenuation_entry = tk.Entry(filter_frame, width=10)
attenuation_entry.grid(row=7, column=1)

wavelet_family_label = tk.Label(filter_frame, text="Wavelet Family:")
wavelet_family_label.grid(row=8, column=0)

wavelet_family_var = tk.StringVar()
wavelet_family_var.trace("w", update_wavelet_options)
wavelet_family_optionmenu = tk.OptionMenu(filter_frame,
                                          wavelet_family_var,
                                          "")
wavelet_family_optionmenu.grid(row=8, column=1)

wavelet_var = tk.StringVar()
wavelet_optionmenu = tk.OptionMenu(filter_frame,
                                   wavelet_var,
                                   "")
wavelet_label = tk.Label(filter_frame, text="Wavelet:")
wavelet_label.grid(row=9, column=0)

wavelet_optionmenu.grid(row=9, column=1)

level_label = tk.Label(filter_frame, text="Level of Decomposition:")
level_label.grid(row=10, column=0)

level_entry = tk.Entry(filter_frame, width=5)
level_entry.grid(row=10, column=1)

level_recon_label = tk.Label(filter_frame, text="Level of Reconstruction:")
level_recon_label.grid(row=11, column=0)

level_recon_entry = tk.Entry(filter_frame, width=5)
level_recon_entry.grid(row=11, column=1)

threshold_method_label = tk.Label(filter_frame, text="Thresholding Method:")
threshold_method_label.grid(row=12,column=0)

threshold_method_var = tk.StringVar(value="Hard")
threshold_method_optionmenu = tk.OptionMenu(filter_frame,
                                            threshold_method_var,
                                            "Hard",
                                            "Soft")
threshold_method_optionmenu.grid(row=12,column=1)

threshold_value_label = tk.Label(filter_frame, text="Threshold Value:")
threshold_value_label.grid(row=13,column=0)

threshold_value_entry = tk.Entry(filter_frame,width=10)
threshold_value_entry.grid(row=13,column=1)

convert_button = tk.Button(root, text="Process Files", command=convert_files)
convert_button.pack()

select_csv_files_button = tk.Button(root, text="Select CSV Files to Plot", command=select_csv_files)
select_csv_files_button.pack()



filter_type_var = tk.StringVar(value="Low Pass")
filter_type_var.trace("w", on_filter_type_change)
low_pass_rb = tk.Radiobutton(filter_frame,
                             text="Low Pass",
                             variable=filter_type_var,
                             value="Low Pass")
low_pass_rb.grid(row=3,column=1)

high_pass_rb = tk.Radiobutton(filter_frame,
                              text="High Pass",
                              variable=filter_type_var,
                              value="High Pass")
high_pass_rb.grid(row=3,column=2)

band_pass_rb = tk.Radiobutton(filter_frame,
                              text="Band Pass",
                              variable=filter_type_var,
                              value="Band Pass")
band_pass_rb.grid(row=3,column=3)

update_wavelet_family_options()
on_filter_design_change()

root.mainloop()
 