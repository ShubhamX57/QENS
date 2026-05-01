
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import pandas as pd
import glob
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

from pandastable import Table  # pip install pandastable


# =========================
# Globals
# =========================
metadata_full = pd.DataFrame()
metadata_filtered = pd.DataFrame()


# =========================
# Helpers: file scanning
# =========================
def build_metadata(folder_path: str) -> pd.DataFrame:
    records = []
    for file in glob.glob(os.path.join(folder_path, "*.nxspe")):
        name = os.path.basename(file).replace(".nxspe", "")
        if name.startswith("empty"):
            continue

        parts = name.split("_")
        if len(parts) < 4:
            # skip unexpected filenames
            continue

        sample = parts[0]
        temperature = float(parts[1])
        Ei = float(parts[2])
        scattering = parts[3]  # coh / inc

        records.append(
            {
                "filepath": file,
                "sample": sample,
                "temperature": temperature,
                "Ei": Ei,
                "scattering": scattering,
            }
        )

    return pd.DataFrame(records)


def validate_files(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    valid, missing = [], []
    for fp in df["filepath"].tolist():
        if os.path.exists(fp):
            valid.append(fp)
        else:
            missing.append(fp)
    return valid, missing


# =========================
# NXSPE load + plot
# =========================
def load_nxspe_data(file_path: str):
    with h5py.File(file_path, "r") as f:
        root_key = list(f.keys())[0]
        print(f"Detect root key: {root_key}")

        data = f[f"{root_key}/data/data"][()]
        energy = f[f"{root_key}/data/energy"][()]
        angles = f[f"{root_key}/data/polar"][()]
        ei = f[f"{root_key}/NXSPE_info/fixed_energy"][()]
        errors = f[f"{root_key}/data/error"][()]

        return data, energy, angles, ei, errors


def plot_data(data, energy, angles, title):
    plot_ctrl = tk.Toplevel(root)
    plot_ctrl.title(f"Plot Control - {title}")
    plot_ctrl.geometry("320x170")

    q_list = angles.tolist()
    default_idx = len(q_list) - 1

    tk.Label(plot_ctrl, text="Select Q value:").pack(pady=5)

    q_options = [f"{q:.4f} Å⁻¹" for q in q_list]
    q_dropdown = ttk.Combobox(plot_ctrl, values=q_options, state="readonly", width=28)
    q_dropdown.current(default_idx)
    q_dropdown.pack(pady=5)

    def on_confirm():
        selected_index = q_dropdown.current()
        selected_q = q_list[selected_index]

        print(f"Current Index: {selected_index}")
        print(f"Corresponding Q: {selected_q:.4f} Å⁻¹")
        print("--- Framework ready, start plotting ---")

        fig, ax = plt.subplots(figsize=(8, 6))

        # NOTE: keeping your indexing logic as-is; adjust if you want a different mapping
        index = data[:, 0].shape[0] - default_idx + 1

        ax.plot(energy[:-1], data[index, :])
        ax.set_xlabel("w / meV")
        ax.set_ylabel("S(w, q)")
        ax.set_title(f"Q = {selected_q:.4f} in {title}")
        plt.show()

    tk.Button(plot_ctrl, text="Confirm & Plot", command=on_confirm, bg="white").pack(pady=10)


# =========================
# Filtering UI logic
# =========================
def set_path(entry_field):
    path = filedialog.askdirectory()
    if path:
        entry_field.delete(0, tk.END)
        entry_field.insert(0, path)


def refresh_filter_controls():
    """
    After loading metadata_full, set slider ranges + type options.
    """
    global metadata_full

    if metadata_full.empty:
        return

    tmin, tmax = float(metadata_full["temperature"].min()), float(metadata_full["temperature"].max())
    emin, emax = float(metadata_full["Ei"].min()), float(metadata_full["Ei"].max())

    # set slider ranges
    temp_min_scale.configure(from_=tmin, to=tmax)
    temp_max_scale.configure(from_=tmin, to=tmax)
    ei_min_scale.configure(from_=emin, to=emax)
    ei_max_scale.configure(from_=emin, to=emax)

    # defaults to full range
    temp_min_var.set(tmin)
    temp_max_var.set(tmax)
    ei_min_var.set(emin)
    ei_max_var.set(emax)

    # type options
    types = sorted(metadata_full["scattering"].unique().tolist())
    type_combo["values"] = types
    if types:
        type_var.set(types[0])

    update_filter_labels()
    apply_filter()


def update_filter_labels():
    temp_label_var.set(f"Temp: {temp_min_var.get():.3g} → {temp_max_var.get():.3g}")
    ei_label_var.set(f"Ei: {ei_min_var.get():.3g} → {ei_max_var.get():.3g}")


def apply_filter(*_):
    """
    Applies current slider/type settings to metadata_full and redraws the table.
    """
    global metadata_full, metadata_filtered, pt

    if metadata_full.empty:
        metadata_filtered = pd.DataFrame()
        pt.model.df = metadata_filtered
        pt.redraw()
        count_var.set("Matching files: 0")
        return

    # enforce min<=max (in case user drags past each other)
    t_lo, t_hi = sorted([float(temp_min_var.get()), float(temp_max_var.get())])
    e_lo, e_hi = sorted([float(ei_min_var.get()), float(ei_max_var.get())])
    scatt = type_var.get()

    filtered = metadata_full[
        metadata_full["temperature"].between(t_lo, t_hi)
        & metadata_full["Ei"].between(e_lo, e_hi)
        & (metadata_full["scattering"] == scatt)
    ].copy()

    metadata_filtered = filtered

    pt.model.df = metadata_filtered
    pt.redraw()

    count_var.set(f"Matching files: {len(metadata_filtered)}")

    # optional: warn about missing paths
    if not metadata_filtered.empty:
        _, missing = validate_files(metadata_filtered)
        if missing:
            print(f"Missing files: {len(missing)}")


def get_path():
    """
    Load folder -> build metadata_full -> init filter controls -> show filtered in table.
    """
    global metadata_full, pt

    folder_path = txt_path.get().strip()
    print(f"Current route is: {folder_path}")

    if not folder_path or not os.path.exists(folder_path):
        messagebox.showerror("Path error", "Folder path does not exist.")
        return

    metadata_full = build_metadata(folder_path)

    if metadata_full.empty:
        pt.model.df = pd.DataFrame()
        pt.redraw()
        count_var.set("Matching files: 0")
        messagebox.showinfo("No data", "No valid .nxspe files found (or all were filtered out as empty/invalid names).")
        return

    print("Metadata loaded.")
    refresh_filter_controls()


# =========================
# Table click handler
# =========================
def handle_row_click(event):
    """
    Double click / click release: load selected row filepath and plot.
    """
    global metadata_filtered

    try:
        row_clicked = pt.get_row_clicked(event)
        if row_clicked < 0:
            return

        if metadata_filtered.empty:
            return

        selected_row = pt.model.df.iloc[row_clicked]
        file_path = selected_row["filepath"]

        print(f"\n[Loading file...] {selected_row['sample']}...")

        if not os.path.exists(file_path):
            messagebox.showerror("Missing file", f"File not found:\n{file_path}")
            return

        data, energy, angles, ei, errors = load_nxspe_data(file_path)

        print(f"Succeed {selected_row['sample']}")
        print(f"data dimension: {data.shape} (Angles x Energy)")
        print(f"Ei: {ei[0] if isinstance(ei, np.ndarray) else ei}")

        plot_data(data, energy, angles, selected_row["sample"])

    except Exception as e:
        print(f"Fail: {e}")
        messagebox.showerror("Fail", str(e))


# =========================
# GUI
# =========================
root = tk.Tk()
root.title("QUENS GUI")
root.geometry("1000x700")

lbl_title = tk.Label(root, text="QUENS data analysis system", font=("Arial", 14))
lbl_title.pack(pady=10)

# Path row
path_frame = tk.Frame(root)
path_frame.pack(fill="x", padx=10, pady=5)

txt_path = tk.Entry(path_frame, width=120)
txt_path.pack(side="left", padx=(0, 8))

btn_get_path = tk.Button(path_frame, text="Select folder", command=lambda: set_path(txt_path))
btn_get_path.pack(side="left", padx=(0, 8))

btn_run = tk.Button(path_frame, text="Load", command=get_path, bg="white")
btn_run.pack(side="left")

# Filter controls
filter_frame = tk.LabelFrame(root, text="Filter (range keeps values; wider values are excluded)", padx=10, pady=10)
filter_frame.pack(fill="x", padx=10, pady=5)

temp_min_var = tk.DoubleVar(value=0.0)
temp_max_var = tk.DoubleVar(value=0.0)
ei_min_var = tk.DoubleVar(value=0.0)
ei_max_var = tk.DoubleVar(value=0.0)

temp_label_var = tk.StringVar(value="Temp: -")
ei_label_var = tk.StringVar(value="Ei: -")

# Temp scales
tk.Label(filter_frame, textvariable=temp_label_var).grid(row=0, column=0, sticky="w")
temp_min_scale = ttk.Scale(filter_frame, orient="horizontal", variable=temp_min_var, command=lambda _: (update_filter_labels(), apply_filter()))
temp_max_scale = ttk.Scale(filter_frame, orient="horizontal", variable=temp_max_var, command=lambda _: (update_filter_labels(), apply_filter()))
temp_min_scale.grid(row=1, column=0, sticky="ew", padx=(0, 10))
temp_max_scale.grid(row=2, column=0, sticky="ew", padx=(0, 10))

# Ei scales
tk.Label(filter_frame, textvariable=ei_label_var).grid(row=0, column=1, sticky="w")
ei_min_scale = ttk.Scale(filter_frame, orient="horizontal", variable=ei_min_var, command=lambda _: (update_filter_labels(), apply_filter()))
ei_max_scale = ttk.Scale(filter_frame, orient="horizontal", variable=ei_max_var, command=lambda _: (update_filter_labels(), apply_filter()))
ei_min_scale.grid(row=1, column=1, sticky="ew")
ei_max_scale.grid(row=2, column=1, sticky="ew")

# Type dropdown
type_var = tk.StringVar(value="")
tk.Label(filter_frame, text="Type:").grid(row=3, column=0, sticky="w", pady=(8, 0))
type_combo = ttk.Combobox(filter_frame, textvariable=type_var, state="readonly", width=20)
type_combo.grid(row=3, column=0, sticky="w", pady=(8, 0), padx=(50, 0))
type_combo.bind("<<ComboboxSelected>>", apply_filter)

# Matching label
count_var = tk.StringVar(value="Matching files: 0")
tk.Label(filter_frame, textvariable=count_var).grid(row=3, column=1, sticky="e", pady=(8, 0))

filter_frame.columnconfigure(0, weight=1)
filter_frame.columnconfigure(1, weight=1)

# Table
frame_table = tk.Frame(root)
frame_table.pack(fill="both", expand=True, padx=10, pady=10)

pt = Table(frame_table, dataframe=pd.DataFrame(), showtoolbar=True, showstatusbar=True)
pt.show()

# single click release (your original binding). If you truly want double-click, change to "<Double-Button-1>"
pt.bind("<ButtonRelease-1>", handle_row_click)

root.mainloop()

# Notes:
# - If nothing shows, check the folder route (Windows vs WSL format).
# - Filenames must be sample_temperature_Ei_scattering.nxspe (underscore-separated).