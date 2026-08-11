# -*- coding: utf-8 -*-
"""
Last updated: Mon, Oct 01, 2025, 07:00
@Author: Michal Kacper Bialobrzewski

First publication release of the Automated Single-Channel Fluorescence Confocal Image Analysis Workflow for Fiji/ImageJ (SChimanal).

This release corresponds to software version 1.0.0 archived in RepOD: https://doi.org/10.18150/2K3PB9

Please cite the following paper when using or adapting this code: https://doi.org/10.64898/2026.07.16.738963.

Who is it for?
Designed for bench scientists and imaging specialists who need reproducible, high-throughput quantification of single-color fluorescence data—especially in the context of biomolecular condensate research. It's also ideal for computational biologists seeking ready-to-analyze tables and figures without the hassle of manually operating ImageJ/Fiji.

Setup Instructions: (coding done in utf-8)

a. Install the bundled Fiji.app 1.54p (https://fiji.sc/).
b. Before analyzing condensates, enable the following measurement options in Fiji: Area, Standard Deviation, Min & Max, Area Fraction, and Perimeter.
In Analyze Particles, check: Display Results, Clear Results, Summarize, Add to Manager, Overlay.
c. Specify the path to your Fiji executable in lines 227 e.g.:'C:/../fiji-win64/Fiji.app/ImageJ-win64.exe'
d. Set the working directory in line 1061, e.g.:'C:/../'
e. Choose thresholding models from the 19 available in Fiji/ImageJ by editing line 1062.
f. To simplify image filenames, define removable patterns in line 39, e.g. 'pH 7', '150 mM NaCl'.
"""

import os
import cv2
import numpy as np
import subprocess
import shutil
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
import re
import logging
from pathlib import Path
import gc

def shorten_file_name(file_name):

    patterns_to_remove = [        
        'pH 7-2', 
    ]
    
    new_name = file_name
    for pattern in patterns_to_remove:
        new_name = new_name.replace(pattern, '')
    
    new_name = re.sub(r'\s+', ' ', new_name)  
    new_name = new_name.strip()               
    
    new_name = re.sub(r'[<>:"/\\|?*]', '', new_name)
    
    return new_name

def shorten_file_names_in_folder(folder_path):

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    folder = Path(folder_path)
    
    if not folder.is_dir():
        logging.error(f"The direction does not exsists: {folder_path}")
        return
    
    for file_path in folder.iterdir():
        if file_path.suffix.lower() not in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.lsm']:
            continue  
        
        original_name = file_path.name
        new_name = shorten_file_name(original_name)
        
        if new_name == original_name:
            continue
        
        new_file_path = folder / new_name
        
        counter = 1
        base_name, ext = os.path.splitext(new_name)
        while new_file_path.exists():
            new_name = f"{base_name}_{counter}{ext}"
            new_file_path = folder / new_name
            counter += 1
        
        try:
            file_path.rename(new_file_path)
        except Exception as e:
            logging.error(f"Could not change the name '{original_name}' -> '{new_name}': {e}")

def prepare_data(output_csv_paths):
    X = []
    y = []
    labels = []
    
    for model, csv_path in output_csv_paths.items():
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)

            if 'Diameter' not in df.columns and 'Perim.' in df.columns:
                df['Diameter'] = df['Perim.'] / np.pi  
            
            if 'StdDev' not in df.columns:
                print(f"Column 'StdDev' missing in file: {csv_path}")
                continue  
            
            df['Diam from Area'] = 2 * np.sqrt(df['Area'] / np.pi)
            df['Error Diam from Area'] = (1 / np.sqrt(df['Area'] * np.pi)) * df['StdDev']
            
            X.append(df[['Area', 'Diameter', 'Diam from Area']].mean().values)
            y.append(model)
            labels.append(model)
        else:
            print(f"File {csv_path} does not exist.")
    
    if not X:  
        print("No valid data was found. Returning None.")
        return None, None, None

    X = np.array(X)
    
    y_encoded = pd.factorize(np.asarray(y))[0]
    return X, y_encoded, labels

def run_imagej_analysis(image_path, output_dir, models):

    import os, textwrap, subprocess

    image_path = image_path.replace("\\", "/")
    output_dir = output_dir.replace("\\", "/")
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]

    macro = textwrap.dedent(f"""
        setBatchMode(true);
        setOption("DisableUndo", true);
        call("ij.Prefs.set", "png.compression", 1);
        run("Close All");

        open("{image_path}");
        if (bitDepth!=8) run("8-bit");

        // ---------- kanały ------------------------------------------------
        getDimensions(w, h, c, z, t);
        if (c>1) {{
            run("Split Channels");

            titles = getList("image.titles");
            found=false;
            for (i=0;i<titles.length;i++) {{
                lc = toLowerCase(titles[i]);
                if (indexOf(lc,"c1")!=-1 || indexOf(lc,"ch1")!=-1 ||
                    indexOf(lc,"channel 1")!=-1) {{
                    selectWindow(titles[i]); rename("C1");
                    found=true; break;
                }}
            }}
            if (!found) {{
                selectWindow(titles[0]); rename("C1");
            }}
        }} else {{
            rename("C1");
        }}

        run("Gaussian Blur...", "sigma=2");
        run("Enhance Contrast", "saturated=0.35 normalize");
        run("Smooth");
        run("Despeckle");
        saveAs("png", "{output_dir}/{base}_original.png");

        getStatistics(a, m, minV, maxV, sd, hist);
        run("Clear Results");
        for (i=0; i<hist.length; i++) {{
            setResult("Value", i, minV + i*(maxV-minV)/hist.length);
            setResult("Count", i, hist[i]);
        }}
        saveAs("Results", "{output_dir}/intensity-histogram.csv");
        run("Clear Results");

        run("Duplicate...", "title=Processed");
    """)

    for m in models:
        macro += textwrap.dedent(f"""
            selectWindow("Processed");
            run("Duplicate...", "title=Processed_{m}");
            selectWindow("Processed_{m}");
            setAutoThreshold("{m}");
            run("Convert to Mask");
            run("Make Binary");
            run("Watershed");

            // standard
            run("Analyze Particles...", "size=0.1-Infinity show=Overlay exclude clear add");
            roiManager("Show All with labels");
            saveAs("png", "{output_dir}/{base}_threshold_{m}.png");
            saveAs("Results", "{output_dir}/results_{m}.csv");
            roiManager("Reset"); run("Clear Results");

            // edges
            run("Analyze Particles...", "size=0.1-Infinity show=Overlay clear add");
            roiManager("Show All with labels");
            saveAs("png", "{output_dir}/{base}_edges_{m}.png");
            saveAs("Results", "{output_dir}/edges_{m}.csv");
            roiManager("Reset"); run("Clear Results");

            close("Processed_{m}");
        """)

    macro += """
        close("Processed");
        run("Close All");
        setBatchMode(false);
    """

    macro_path = os.path.join(output_dir, "macro_analysis.ijm").replace("\\", "/")
    with open(macro_path, "w", encoding="utf-8") as f:
        f.write(macro)

    try:
        subprocess.run(
            ["C:/../fiji-win64/Fiji.app/ImageJ-win64.exe",
             "-batch", macro_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=600
        )
    except subprocess.TimeoutExpired:
        print("✗ Fiji macro timed out (600 s)")

    created = [os.path.join(output_dir, f"{base}_original.png"),
               os.path.join(output_dir, "intensity-histogram.csv"),
               macro_path]
    for m in models:
        created += [
            os.path.join(output_dir, f"{base}_threshold_{m}.png"),
            os.path.join(output_dir, f"results_{m}.csv"),
            os.path.join(output_dir, f"{base}_edges_{m}.png"),
            os.path.join(output_dir, f"edges_{m}.csv")
        ]
    return created

def round_significant(value, sig_digits=2):
    if value == 0:
        return 0
    return round(value, -int(np.floor(np.log10(abs(value))) - (sig_digits - 1)))

def round_dataframe_to_significant(df, columns, sig_digits=2):
    for col in columns:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: round_significant(x, sig_digits))
    return df

def calculate_statistics_and_save(output_csv_paths, output_file_path):
    data = []
    for model, csv_path in output_csv_paths.items():
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)

            if 'Diameter' not in df.columns and 'Perim.' in df.columns:
                df['Diameter'] = df['Perim.'] / np.pi

            median_area = df['Area'].median()
            q1_area = df['Area'].quantile(0.25)
            q3_area = df['Area'].quantile(0.75)
            median_diameter = df['Diameter'].median()
            q1_diameter = df['Diameter'].quantile(0.25)
            q3_diameter = df['Diameter'].quantile(0.75)
            median_diam_from_area = (2 * np.sqrt(df['Area'] / np.pi)).median()
            q1_diam_from_area = (2 * np.sqrt(df['Area'] / np.pi)).quantile(0.25)
            q3_diam_from_area = (2 * np.sqrt(df['Area'] / np.pi)).quantile(0.75)
            count = len(df)

            data.append({
                'Model': model,
                'Median Area': round_significant(median_area),
                'Q1 Area': round_significant(q1_area),
                'Q3 Area': round_significant(q3_area),
                'Median Diameter': round_significant(median_diameter),
                'Q1 Diameter': round_significant(q1_diameter),
                'Q3 Diameter': round_significant(q3_diameter),
                'Median Diam from Area': round_significant(median_diam_from_area),
                'Q1 Diam from Area': round_significant(q1_diam_from_area),
                'Q3 Diam from Area': round_significant(q3_diam_from_area),
                'Count': count,
                'Sqrt Count': round_significant(np.sqrt(count))
            })
    
    all_models_df = pd.DataFrame(data)

    with pd.ExcelWriter(output_file_path) as writer:
        all_models_df.to_excel(writer, sheet_name='Model Statistics', index=False)

        overall_stats = {
            'Median Area': round_significant(all_models_df['Median Area'].median()),
            'Q1 Area': round_significant(all_models_df['Q1 Area'].median()),
            'Q3 Area': round_significant(all_models_df['Q3 Area'].median()),
            'Median Diameter': round_significant(all_models_df['Median Diameter'].median()),
            'Q1 Diameter': round_significant(all_models_df['Q1 Diameter'].median()),
            'Q3 Diameter': round_significant(all_models_df['Q3 Diameter'].median()),
            'Median Diam from Area': round_significant(all_models_df['Median Diam from Area'].median()),
            'Q1 Diam from Area': round_significant(all_models_df['Q1 Diam from Area'].median()),
            'Q3 Diam from Area': round_significant(all_models_df['Q3 Diam from Area'].median()),
            'Counts': round_significant(all_models_df['Count'].mean()),
            'STD Counts': round_significant(all_models_df['Count'].std())
        }
        overall_stats_df = pd.DataFrame([overall_stats])
        overall_stats_df.to_excel(writer, sheet_name='Overall Statistics', index=False)

def calculate_edge_statistics_and_save(edge_csv_paths, output_file_path):
    data = []
    for model, csv_path in edge_csv_paths.items():
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)

            if 'Diameter' not in df.columns and 'Perim.' in df.columns:
                df['Diameter'] = df['Perim.'] / np.pi

            median_area = df['Area'].median()
            q1_area = df['Area'].quantile(0.25)
            q3_area = df['Area'].quantile(0.75)
            median_diameter = df['Diameter'].median()
            q1_diameter = df['Diameter'].quantile(0.25)
            q3_diameter = df['Diameter'].quantile(0.75)
            median_diam_from_area = (2 * np.sqrt(df['Area'] / np.pi)).median()
            q1_diam_from_area = (2 * np.sqrt(df['Area'] / np.pi)).quantile(0.25)
            q3_diam_from_area = (2 * np.sqrt(df['Area'] / np.pi)).quantile(0.75)
            count = len(df)

            data.append({
                'Model': model,
                'Median Area': round_significant(median_area),
                'Q1 Area': round_significant(q1_area),
                'Q3 Area': round_significant(q3_area),
                'Median Diameter': round_significant(median_diameter),
                'Q1 Diameter': round_significant(q1_diameter),
                'Q3 Diameter': round_significant(q3_diameter),
                'Median Diam from Area': round_significant(median_diam_from_area),
                'Q1 Diam from Area': round_significant(q1_diam_from_area),
                'Q3 Diam from Area': round_significant(q3_diam_from_area),
                'Count': count,
                'Sqrt Count': round_significant(np.sqrt(count))
            })
    
    all_models_df = pd.DataFrame(data)

    with pd.ExcelWriter(output_file_path) as writer:
        all_models_df.to_excel(writer, sheet_name='Model Statistics', index=False)

        overall_stats = {
            'Median Area': round_significant(all_models_df['Median Area'].median()),
            'Q1 Area': round_significant(all_models_df['Q1 Area'].median()),
            'Q3 Area': round_significant(all_models_df['Q3 Area'].median()),
            'Median Diameter': round_significant(all_models_df['Median Diameter'].median()),
            'Q1 Diameter': round_significant(all_models_df['Q1 Diameter'].median()),
            'Q3 Diameter': round_significant(all_models_df['Q3 Diameter'].median()),
            'Median Diam from Area': round_significant(all_models_df['Median Diam from Area'].median()),
            'Q1 Diam from Area': round_significant(all_models_df['Q1 Diam from Area'].median()),
            'Q3 Diam from Area': round_significant(all_models_df['Q3 Diam from Area'].median()),
            'Counts': round_significant(all_models_df['Count'].mean()),
            'STD Counts': round_significant(all_models_df['Count'].std())
        }
        overall_stats_df = pd.DataFrame([overall_stats])
        overall_stats_df.to_excel(writer, sheet_name='Overall Statistics', index=False)

def merge_results_to_excel(image_name, output_csv_paths, models, output_excel_path, stats_file_path, edge_stats_file_path):
    with pd.ExcelWriter(output_excel_path) as writer:
        stats_df = pd.read_excel(stats_file_path, sheet_name='Model Statistics')
        edge_stats_df = pd.read_excel(edge_stats_file_path, sheet_name='Model Statistics')

        for model in models:
            csv_path = output_csv_paths.get(model, None)
            if csv_path and os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                if 'Diameter' not in df.columns and 'Perim.' in df.columns:
                    df['Diameter'] = df['Perim.'] / np.pi
                df['Diam from Area'] = 2 * np.sqrt(df['Area'] / np.pi)
                df.to_excel(writer, sheet_name=model, index=False)

        summary_data = []
        for model in models:
            model_stats = stats_df[stats_df['Model'] == model]
            edge_model_stats = edge_stats_df[edge_stats_df['Model'] == model]
            if not model_stats.empty:
                summary_data.append({
                    'Model': model,
                    'Median Area': round_significant(model_stats['Median Area'].values[0]),
                    'Q1 Area': round_significant(model_stats['Q1 Area'].values[0]),
                    'Q3 Area': round_significant(model_stats['Q3 Area'].values[0]),
                    'Median Diameter': round_significant(model_stats['Median Diameter'].values[0]),
                    'Q1 Diameter': round_significant(model_stats['Q1 Diameter'].values[0]),
                    'Q3 Diameter': round_significant(model_stats['Q3 Diameter'].values[0]),
                    'Median Diam from Area': round_significant(model_stats['Median Diam from Area'].values[0]),
                    'Q1 Diam from Area': round_significant(model_stats['Q1 Diam from Area'].values[0]),
                    'Q3 Diam from Area': round_significant(model_stats['Q3 Diam from Area'].values[0]),
                    'Count': model_stats['Count'].values[0],
                    'Sqrt Count': round_significant(np.sqrt(model_stats['Count'].values[0]))
                })

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

def plot_statistics(stats_file_path, edge_stats_file_path, output_plot_path):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # Load statistics data
    stats_df = pd.read_excel(stats_file_path, sheet_name='Model Statistics')
    overall_stats = pd.read_excel(stats_file_path, sheet_name='Overall Statistics').iloc[0]

    edge_stats_df = pd.read_excel(edge_stats_file_path, sheet_name='Model Statistics')
    edge_overall_stats = pd.read_excel(edge_stats_file_path, sheet_name='Overall Statistics').iloc[0]


    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 18  # General font size for axis labels
    axis_label_fontsize = 18  # Font size for axis titles
    tick_label_fontsize = 16  # Font size for tick labels

    # Calculate SD for Counts
    count_sd = stats_df['Count'].std()
    overall_count = overall_stats['Counts']

    # Create a figure with 3 subplots
    plt.figure(figsize=(18, 6))
    
    # 1. Area subplot
    plt.subplot(1, 3, 1)
    plt.errorbar(stats_df['Model'], stats_df['Median Area'],
                 yerr=[stats_df['Median Area'] - stats_df['Q1 Area'], stats_df['Q3 Area'] - stats_df['Median Area']],
                 fmt='o', color='blue', label='Median Area', markersize=8)  # Increased marker size
    plt.errorbar(edge_stats_df['Model'], edge_stats_df['Median Area'],
                 yerr=[edge_stats_df['Median Area'] - edge_stats_df['Q1 Area'], edge_stats_df['Q3 Area'] - edge_stats_df['Median Area']],
                 fmt='s', color='orange', label='Edge Median Area', markersize=8)  # Increased marker size
    plt.axhline(y=overall_stats['Median Area'], color='r', linestyle='--', label='Overall Median Area')
    plt.axhline(y=overall_stats['Median Area'] + (overall_stats['Q3 Area'] - overall_stats['Median Area']), color='gray', linestyle='--', label='+1Q Area')
    plt.axhline(y=overall_stats['Median Area'] - (overall_stats['Median Area'] - overall_stats['Q1 Area']), color='gray', linestyle='--')
    plt.axhline(y=overall_stats['Median Area'] + 2 * (overall_stats['Q3 Area'] - overall_stats['Median Area']), color='black', linestyle='--', label='+2Q Area')
    plt.axhline(y=overall_stats['Median Area'] - 2 * (overall_stats['Median Area'] - overall_stats['Q1 Area']), color='black', linestyle='--')
    plt.xlabel('Threshold Models', fontsize=axis_label_fontsize, labelpad=20)
    plt.ylabel('Area (µm²)', fontsize=axis_label_fontsize, labelpad=20)
    plt.xticks(rotation=90, fontsize=tick_label_fontsize)
    plt.yticks(fontsize=tick_label_fontsize)
    
    # 2. Diameter subplot
    plt.subplot(1, 3, 2)
    plt.errorbar(stats_df['Model'], stats_df['Median Diameter'],
                 yerr=[stats_df['Median Diameter'] - stats_df['Q1 Diameter'], stats_df['Q3 Diameter'] - stats_df['Median Diameter']],
                 fmt='o', color='blue', label='Median Diameter', markersize=8)  # Increased marker size
    plt.errorbar(edge_stats_df['Model'], edge_stats_df['Median Diameter'],
                 yerr=[edge_stats_df['Median Diameter'] - edge_stats_df['Q1 Diameter'], edge_stats_df['Q3 Diameter'] - edge_stats_df['Median Diameter']],
                 fmt='s', color='orange', label='Edge Median Diameter', markersize=8)  # Increased marker size
    plt.axhline(y=overall_stats['Median Diameter'], color='r', linestyle='--', label='Overall Median Diameter')
    plt.axhline(y=overall_stats['Median Diameter'] + (overall_stats['Q3 Diameter'] - overall_stats['Median Diameter']), color='gray', linestyle='--', label='+1Q Diameter')
    plt.axhline(y=overall_stats['Median Diameter'] - (overall_stats['Median Diameter'] - overall_stats['Q1 Diameter']), color='gray', linestyle='--')
    plt.axhline(y=overall_stats['Median Diameter'] + 2 * (overall_stats['Q3 Diameter'] - overall_stats['Median Diameter']), color='black', linestyle='--', label='+2Q Diameter')
    plt.axhline(y=overall_stats['Median Diameter'] - 2 * (overall_stats['Median Diameter'] - overall_stats['Q1 Diameter']), color='black', linestyle='--')
    plt.xlabel('Threshold Models', fontsize=axis_label_fontsize, labelpad=20)
    plt.ylabel('Perimeter (µm)', fontsize=axis_label_fontsize, labelpad=20)
    plt.xticks(rotation=90, fontsize=tick_label_fontsize)
    plt.yticks(fontsize=tick_label_fontsize)
    
    # 3. Counts subplot
    plt.subplot(1, 3, 3)
    plt.errorbar(stats_df['Model'], stats_df['Count'],
                 yerr=count_sd, fmt='o', color='blue', label='Counts', markersize=8)  # Increased marker size
    plt.errorbar(edge_stats_df['Model'], edge_stats_df['Count'],
                 yerr=edge_stats_df['Count'].std(), fmt='s', color='orange', label='Edge Counts', markersize=8)  # Increased marker size
    plt.axhline(y=overall_count, color='r', linestyle='--', label='Overall Median Count')
    plt.axhline(y=overall_count + count_sd, color='gray', linestyle='--', label='+1SD Count')
    plt.axhline(y=overall_count - count_sd, color='gray', linestyle='--')
    plt.axhline(y=overall_count + 2 * count_sd, color='black', linestyle='--', label='+2SD Count')
    plt.axhline(y=overall_count - 2 * count_sd, color='black', linestyle='--')
    plt.xlabel('Threshold Models', fontsize=axis_label_fontsize, labelpad=20)
    plt.ylabel('Counts (a.u.)', fontsize=axis_label_fontsize, labelpad=20)
    plt.xticks(rotation=90, fontsize=tick_label_fontsize)
    plt.yticks(fontsize=tick_label_fontsize)

    handles, labels = plt.gca().get_legend_handles_labels()
    plt.figlegend(handles, labels, loc='lower center', ncol=len(labels), fontsize='medium', bbox_to_anchor=(0.5, -0.07))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
    #plt.show()
    plt.close()

def predict_best_threshold_model(output_csv_paths):
    data = []
    
    for model, csv_path in output_csv_paths.items():
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            
            # Add Diameter column if it doesn't exist
            if 'Diameter' not in df.columns and 'Perim.' in df.columns:
                df['Diameter'] = df['Perim.'] / np.pi  # Calculate diameter from perimeter
            
            # Check for 'Area' column
            if 'Area' not in df.columns:
                print(f"Column 'Area' missing in file: {csv_path}")
                continue

            # Convert 'Area' to numeric and handle NaN values
            df['Area'] = pd.to_numeric(df['Area'], errors='coerce')
            if df['Area'].isnull().all():
                print(f"All 'Area' values are NaN after conversion in file: {csv_path}")
                continue

            # Calculate median values for Area, Diameter, and Diam from Area
            median_area = df['Area'].median()
            q1_area = df['Area'].quantile(0.25)
            q3_area = df['Area'].quantile(0.75)
            
            median_diameter = df['Diameter'].median()
            q1_diameter = df['Diameter'].quantile(0.25)
            q3_diameter = df['Diameter'].quantile(0.75)
            
            diam_from_area = 2 * np.sqrt(df['Area'] / np.pi)
            median_diam_from_area = diam_from_area.median()
            q1_diam_from_area = diam_from_area.quantile(0.25)
            q3_diam_from_area = diam_from_area.quantile(0.75)

            count = len(df)

            data.append({
                'Model': model,
                'Median Area': median_area,
                'Q1 Area': q1_area,
                'Q3 Area': q3_area,
                'Median Diameter': median_diameter,
                'Q1 Diameter': q1_diameter,
                'Q3 Diameter': q3_diameter,
                'Diam from Area': median_diam_from_area,
                'Q1 Diam from Area': q1_diam_from_area,
                'Q3 Diam from Area': q3_diam_from_area,
                'Count': count
            })
        else:
            print(f"File {csv_path} does not exist.")
    
    # Create a DataFrame for all models
    all_models_df = pd.DataFrame(data)

    if all_models_df.empty:
        print("No valid data found for prediction.")
        return [], [], [], None

    # Calculate overall medians and standard deviations for comparisons
    overall_median_area = all_models_df['Median Area'].median()
    overall_median_diameter = all_models_df['Median Diameter'].median()
    overall_median_diam_from_area = all_models_df['Diam from Area'].median()
    overall_median_count = all_models_df['Count'].median()

    sd_area = all_models_df['Median Area'].std()
    sd_diameter = all_models_df['Median Diameter'].std()
    sd_diam_from_area = all_models_df['Diam from Area'].std()
    sd_count = all_models_df['Count'].std()

    # Helper function to check how close a model is to the overall median values
    def classify_model(row):
        within_1sd = (
            abs(row['Median Area'] - overall_median_area) <= sd_area and
            abs(row['Median Diameter'] - overall_median_diameter) <= sd_diameter and
            abs(row['Diam from Area'] - overall_median_diam_from_area) <= sd_diam_from_area and
            abs(row['Count'] - overall_median_count) <= sd_count
        )
        if within_1sd:
            return "Within 1 SD"
        
        within_2sd = (
            abs(row['Median Area'] - overall_median_area) <= 2 * sd_area and
            abs(row['Median Diameter'] - overall_median_diameter) <= 2 * sd_diameter and
            abs(row['Diam from Area'] - overall_median_diam_from_area) <= 2 * sd_diam_from_area and
            abs(row['Count'] - overall_median_count) <= 2 * sd_count
        )
        if within_2sd:
            return "Within 2 SD"
        
        within_3sd = (
            abs(row['Median Area'] - overall_median_area) <= 3 * sd_area and
            abs(row['Median Diameter'] - overall_median_diameter) <= 3 * sd_diameter and
            abs(row['Diam from Area'] - overall_median_diam_from_area) <= 3 * sd_diam_from_area and
            abs(row['Count'] - overall_median_count) <= 3 * sd_count
        )
        if within_3sd:
            return "Within 3 SD"
        
        return "Outside 3 SD"

    # Apply classification to each model
    all_models_df['Model Category'] = all_models_df.apply(classify_model, axis=1)

    # Extract models within each category
    best_models_within_1sd = all_models_df[all_models_df['Model Category'] == 'Within 1 SD']['Model'].tolist()
    models_within_2sd = all_models_df[all_models_df['Model Category'] == 'Within 2 SD']['Model'].tolist()
    models_within_3sd = all_models_df[all_models_df['Model Category'] == 'Within 3 SD']['Model'].tolist()

    # Return results along with statistics for saving and further analysis
    return best_models_within_1sd, models_within_2sd, models_within_3sd, {
        'Overall Mean Area': overall_median_area,
        'STD Area': sd_area,
        'Overall Mean Diameter': overall_median_diameter,
        'STD Diameter': sd_diameter,
        'Overall Mean Diam from Area': overall_median_diam_from_area,
        'STD Diam from Area': sd_diam_from_area,
        'Overall Mean Count': overall_median_count,
        'STD Count': sd_count
    }

def save_predictions_to_excel(best_models_within_1sd, models_within_2sd, models_within_3sd, model_stats, statistics_file_path, output_file_path):
    if model_stats is None:
        print("No model statistics available. Skipping predictions saving.")
        return

    stats_df = pd.read_excel(statistics_file_path, sheet_name='Model Statistics')
    
    overall_sd_count = stats_df['Count'].std()

    overall_stats = {
        'Overall Mean Area': round_significant(model_stats['Overall Mean Area'], 2),
        'STD Area': round_significant(model_stats['STD Area'], 2),
        'Overall Mean Diameter': round_significant(model_stats['Overall Mean Diameter'], 2),
        'STD Diameter': round_significant(model_stats['STD Diameter'], 2),
        'Overall Mean Diam from Area': round_significant(model_stats['Overall Mean Diam from Area'], 2),
        'STD Diam from Area': round_significant(model_stats['STD Diam from Area'], 2),
        'Overall Mean Count': round_significant(model_stats['Overall Mean Count'], 2),
        'STD Count': round_significant(overall_sd_count, 2)  # Standard deviation for Counts
    }

    predictions_data = {
        'Model Category': [],
        'Model Name': [],
        'Median Area': [],
        'Q1 Area': [],
        'Q3 Area': [],
        'Median Diameter': [],
        'Q1 Diameter': [],
        'Q3 Diameter': [],
        'Median Diam from Area': [],
        'Q1 Diam from Area': [],
        'Q3 Diam from Area': [],
        'Count': [],
        'Sqrt Count': [],
        'STD Count': []
    }

    def get_model_stats(model_name):
        model_row = stats_df[stats_df['Model'] == model_name]
        if not model_row.empty:
            return model_row.iloc[0]
        return None

    # Fill data for models within 1 SD
    for model in best_models_within_1sd:
        model_stats_row = get_model_stats(model)
        if model_stats_row is not None:
            predictions_data['Model Category'].append('Within 1 SD')
            predictions_data['Model Name'].append(model)
            predictions_data['Median Area'].append(round_significant(model_stats_row['Median Area'], 2))
            predictions_data['Q1 Area'].append(round_significant(model_stats_row['Q1 Area'], 2))
            predictions_data['Q3 Area'].append(round_significant(model_stats_row['Q3 Area'], 2))
            predictions_data['Median Diameter'].append(round_significant(model_stats_row['Median Diameter'], 2))
            predictions_data['Q1 Diameter'].append(round_significant(model_stats_row['Q1 Diameter'], 2))
            predictions_data['Q3 Diameter'].append(round_significant(model_stats_row['Q3 Diameter'], 2))
            predictions_data['Median Diam from Area'].append(round_significant(model_stats_row.get('Median Diam from Area', 0), 2))
            predictions_data['Q1 Diam from Area'].append(round_significant(model_stats_row.get('Q1 Diam from Area', 0), 2))
            predictions_data['Q3 Diam from Area'].append(round_significant(model_stats_row.get('Q3 Diam from Area', 0), 2))
            predictions_data['Count'].append(round_significant(model_stats_row['Count'], 2))
            predictions_data['Sqrt Count'].append(round_significant(np.sqrt(model_stats_row['Count']), 2))
            predictions_data['STD Count'].append(round_significant(overall_sd_count, 2))  # Standard deviation for Count

    # Repeat for models within 2 SD and 3 SD
    for model in models_within_2sd:
        model_stats_row = get_model_stats(model)
        if model_stats_row is not None:
            predictions_data['Model Category'].append('Within 2 SD')
            predictions_data['Model Name'].append(model)
            predictions_data['Median Area'].append(round_significant(model_stats_row['Median Area'], 2))
            predictions_data['Q1 Area'].append(round_significant(model_stats_row['Q1 Area'], 2))
            predictions_data['Q3 Area'].append(round_significant(model_stats_row['Q3 Area'], 2))
            predictions_data['Median Diameter'].append(round_significant(model_stats_row['Median Diameter'], 2))
            predictions_data['Q1 Diameter'].append(round_significant(model_stats_row['Q1 Diameter'], 2))
            predictions_data['Q3 Diameter'].append(round_significant(model_stats_row['Q3 Diameter'], 2))
            predictions_data['Median Diam from Area'].append(round_significant(model_stats_row.get('Median Diam from Area', 0), 2))
            predictions_data['Q1 Diam from Area'].append(round_significant(model_stats_row.get('Q1 Diam from Area', 0), 2))
            predictions_data['Q3 Diam from Area'].append(round_significant(model_stats_row.get('Q3 Diam from Area', 0), 2))
            predictions_data['Count'].append(round_significant(model_stats_row['Count'], 2))
            predictions_data['Sqrt Count'].append(round_significant(np.sqrt(model_stats_row['Count']), 2))
            predictions_data['STD Count'].append(round_significant(overall_sd_count, 2))

    for model in models_within_3sd:
        model_stats_row = get_model_stats(model)
        if model_stats_row is not None:
            predictions_data['Model Category'].append('Within 3 SD')
            predictions_data['Model Name'].append(model)
            predictions_data['Median Area'].append(round_significant(model_stats_row['Median Area'], 2))
            predictions_data['Q1 Area'].append(round_significant(model_stats_row['Q1 Area'], 2))
            predictions_data['Q3 Area'].append(round_significant(model_stats_row['Q3 Area'], 2))
            predictions_data['Median Diameter'].append(round_significant(model_stats_row['Median Diameter'], 2))
            predictions_data['Q1 Diameter'].append(round_significant(model_stats_row['Q1 Diameter'], 2))
            predictions_data['Q3 Diameter'].append(round_significant(model_stats_row['Q3 Diameter'], 2))
            predictions_data['Median Diam from Area'].append(round_significant(model_stats_row.get('Median Diam from Area', 0), 2))
            predictions_data['Q1 Diam from Area'].append(round_significant(model_stats_row.get('Q1 Diam from Area', 0), 2))
            predictions_data['Q3 Diam from Area'].append(round_significant(model_stats_row.get('Q3 Diam from Area', 0), 2))
            predictions_data['Count'].append(round_significant(model_stats_row['Count'], 2))
            predictions_data['Sqrt Count'].append(round_significant(np.sqrt(model_stats_row['Count']), 2))
            predictions_data['STD Count'].append(round_significant(overall_sd_count, 2))

    # Convert to DataFrame
    predictions_df = pd.DataFrame(predictions_data)

    # Save to Excel
    with pd.ExcelWriter(output_file_path) as writer:
        # Save overall statistics
        pd.DataFrame([overall_stats]).to_excel(writer, sheet_name='Overall Statistics', index=False)
        # Save predictions
        predictions_df.to_excel(writer, sheet_name='Machine Predictions', index=False)

def generate_grid(output_dir, threshold_models, grid_size=5, thumbnail_size=(200, 200),
                 padding=30, font_size=40, frame_color=(0, 0, 0), text_color=(0, 0, 0),
                 background_color=(255, 255, 255), frame_thickness=3, side_margin=150, top_margin=150, bottom_margin=150,
                 text_padding=30, scale_factor=1):
    """
    Generates a high-quality grid image comparing the results of different thresholding models with larger thumbnails.
    Includes the original image as the first thumbnail labeled 'Original'.

    Parameters:
        output_dir (str): Directory where the thresholded images are saved.
        threshold_models (list): List of thresholding models used.
        grid_size (int): Number of columns in the grid.
        thumbnail_size (tuple): Size of the thumbnails (width, height) in pixels.
        padding (int): Space between images and around the grid in pixels.
        font_size (int): Size of the font for model names.
        frame_color (tuple): Color of the frame around each thumbnail (R, G, B).
        text_color (tuple): Color of the model name text (R, G, B).
        background_color (tuple): Background color of the grid image (R, G, B).
        frame_thickness (int): Thickness of the frame around the image in pixels.
        side_margin (int): Left and right margin size in pixels.
        top_margin (int): Top margin size in pixels.
        bottom_margin (int): Bottom margin size in pixels.
        text_padding (int): Additional padding between the image and the model name in pixels.
        scale_factor (int): Factor by which to scale the grid image for higher resolution.
    """
    image_info = []

    original_image_path = None
    for file in os.listdir(output_dir):
        if file.endswith("_original.png"):
            original_image_path = os.path.join(output_dir, file)
            break
    if original_image_path:
        image_info.append((original_image_path, 'Original'))

    for model in threshold_models:
        img_files = [f for f in os.listdir(output_dir) if f.endswith(f"_threshold_{model}.png")]
        for img_file in img_files:
            image_path = os.path.join(output_dir, img_file)
            image_info.append((image_path, model))

    num_images = len(image_info)
    num_rows = (num_images + grid_size - 1) // grid_size

    try:
        font = ImageFont.truetype("arial.ttf", font_size * scale_factor)
    except IOError:
        font = ImageFont.load_default()

    thumbnail_width, thumbnail_height = thumbnail_size
    grid_img_width = (thumbnail_width + padding) * grid_size - padding + 2 * side_margin
    grid_img_height = (thumbnail_height + padding + font_size + text_padding) * num_rows + top_margin + bottom_margin

    grid_img_width_scaled = grid_img_width * scale_factor
    grid_img_height_scaled = grid_img_height * scale_factor

    grid_img = Image.new('RGB', (grid_img_width_scaled, grid_img_height_scaled), background_color)

    draw = ImageDraw.Draw(grid_img)

    for i, (img_path, label) in enumerate(image_info):
        try:
            img = Image.open(img_path).convert('RGB')
        except IOError:
            print(f"Nie można otworzyć obrazu: {img_path}")
            continue

        img = add_padding(img, thumbnail_size, background_color)

        img = img.resize(thumbnail_size, Image.Resampling.LANCZOS)

        row = i // grid_size
        col = i % grid_size
        x_offset = col * (thumbnail_width + padding) + side_margin
        y_offset = row * (thumbnail_height + padding + font_size + text_padding) + top_margin

        x_offset_scaled = x_offset * scale_factor
        y_offset_scaled = y_offset * scale_factor

        thumbnail_width_scaled = thumbnail_width * scale_factor
        thumbnail_height_scaled = thumbnail_height * scale_factor
        frame_thickness_scaled = frame_thickness * scale_factor
        text_padding_scaled = text_padding * scale_factor
        font_size_scaled = font_size * scale_factor

        if isinstance(font, ImageFont.FreeTypeFont):
            try:
                font_scaled = ImageFont.truetype("arial.ttf", font_size_scaled)
            except IOError:
                font_scaled = font
        else:
            font_scaled = font


        frame = Image.new('RGB', (thumbnail_width_scaled + 2 * frame_thickness_scaled, 
                                  thumbnail_height_scaled + 2 * frame_thickness_scaled), frame_color)
        frame.paste(img.resize((thumbnail_width_scaled, thumbnail_height_scaled), Image.Resampling.LANCZOS), 
                   (frame_thickness_scaled, frame_thickness_scaled))  # Wyśrodkuj obraz w ramce

        grid_img.paste(frame, (x_offset_scaled, y_offset_scaled))

        model_name = label  

        text_bbox = draw.textbbox((0, 0), model_name, font=font_scaled)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        text_x = x_offset_scaled + (thumbnail_width_scaled - text_width) // 2
        text_y = y_offset_scaled + thumbnail_height_scaled + text_padding_scaled

        shadow_offset = (int(2 * scale_factor), int(2 * scale_factor))
        shadow_color = (255, 255, 255) 
        draw.text((text_x + shadow_offset[0], text_y + shadow_offset[1]), model_name, font=font_scaled, fill=shadow_color)

        draw.text((text_x, text_y), model_name, font=font_scaled, fill=text_color)

    grid_img_path = os.path.join(output_dir, 'grid_threshold_summary.png')
    grid_img.save(grid_img_path, format='PNG', dpi=(300, 300))

    grid_array = np.array(grid_img)
    plt.figure(figsize=(grid_img_width_scaled / 100, grid_img_height_scaled / 100), dpi=300) 
    plt.imshow(grid_array)
    plt.axis('off') 
    #plt.show()
    plt.close()

def add_padding(img, target_size, background_color=(255, 255, 255)):

    img_ratio = img.width / img.height
    target_ratio = target_size[0] / target_size[1]

    if img_ratio > target_ratio:
        new_width = target_size[0]
        new_height = int(new_width / img_ratio)
    else:
        new_height = target_size[1]
        new_width = int(new_height * img_ratio)

    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    new_img = Image.new('RGB', target_size, background_color)
    paste_position = ((target_size[0] - new_width) // 2, (target_size[1] - new_height) // 2)
    new_img.paste(img, paste_position)
    return new_img

def move_files_to_folder(files, folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    for file in files:
        if os.path.exists(file):
            try:
                dest = os.path.join(folder_name, os.path.basename(file))
                shutil.move(file, dest)
            except Exception as e:
                print(f"Failed to move file {file} to {folder_name}: {e}")

def generate_intensity_histogram(output_dir):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from PIL import Image
    import os

    # Setting font and label sizes
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 18  # General font size
    axis_label_fontsize = 18  # Font size for axis labels
    tick_label_fontsize = 16  # Font size for tick labels

    try:
        histogram_data_path = os.path.join(output_dir, 'intensity-histogram.csv')
        if not os.path.exists(histogram_data_path):
            print("Histogram data file not found.")
            return

        df = pd.read_csv(histogram_data_path)
        #print(f"DataFrame shape: {df.shape}")
        #print(f"DataFrame columns: {df.columns.tolist()}")

        if 'Value' not in df.columns or 'Count' not in df.columns:
            print("Expected columns 'Value' and 'Count' not found in histogram data.")
            return

        values = np.repeat(df['Value'], df['Count'].astype(int))
        #print(f"Total number of intensity values: {values.size}")

        values = values[values >= 0]  # Remove negative values
        #print(f"Number of positive intensity values: {values.size}")

        if values.size == 0:
            print("No valid intensity data to plot.")
            return

        # Calculating statistics
        total_pixels = values.size
        mean_value = values.mean()
        median_value = np.median(values)
        mode_value = df['Value'][df['Count'].idxmax()]
        min_value = values.min()
        max_value = values.max()
        std_value = values.std()
        #print(f"Total Pixels: {total_pixels}, Mean: {mean_value:.2f}, Median: {median_value}, Mode: {mode_value}, Min: {min_value}, Max: {max_value}")

        # Finding the original image
        original_image_path = None
        for file in os.listdir(output_dir):
            if file.endswith("_original.png"):
                original_image_path = os.path.join(output_dir, file)
                break

        # Determining the gradient color based on the image's dominant color
        gradient_color = 'red'  # Default color

        if original_image_path and os.path.exists(original_image_path):
            image = Image.open(original_image_path).convert('RGB')
            image_np = np.array(image)

            # Define conditions for colors
            red_pixels = (image_np[:, :, 0] > 200) & (image_np[:, :, 1] < 50) & (image_np[:, :, 2] < 50)
            green_pixels = (image_np[:, :, 1] > 200) & (image_np[:, :, 0] < 50) & (image_np[:, :, 2] < 50)
            yellow_pixels = (image_np[:, :, 0] > 200) & (image_np[:, :, 1] > 200) & (image_np[:, :, 2] < 50)

            if np.any(red_pixels):
                gradient_color = 'orangered'  # Brighter shade of red
            elif np.any(green_pixels):
                gradient_color = 'lime'  # Brighter shade of green
            elif np.any(yellow_pixels):
                gradient_color = 'yellow'
            else:
                print("No red, green, or yellow pixels detected. Gradient set to black -> white.")
                gradient_color = 'white'
        else:
            print("Original image not found. Gradient set to black -> white.")

        # Creating a custom color map from black to the determined color
        cmap = LinearSegmentedColormap.from_list('CustomScale', [(0, 'black'), (1, gradient_color)])

        # Creating the plot
        fig, ax = plt.subplots(figsize=(12, 7))
        sns.set(style='whitegrid')

        # Plotting the histogram in logarithmic scale
        n, bins, patches = ax.hist(values, bins=256, color='white', edgecolor='black', alpha=0.55, log=True)

        # Setting X-axis limits
        ax.set_xlim(0, 260)

        # Setting Y-axis limits
        ax.set_ylim(1000, 1e7)  # Set Y-axis from 1000 to 10^7

        # Setting labels and title
        ax.set_xlabel('Intensity Value', fontsize=axis_label_fontsize)
        ax.set_ylabel('Frequency (log scale)', fontsize=axis_label_fontsize)
        #ax.set_title('Fluorescence Intensity Distribution Histogram')  # Changed title
        ax.tick_params(axis='x', labelsize=tick_label_fontsize)
        ax.tick_params(axis='y', labelsize=tick_label_fontsize)

        # Creating the color gradient
        gradient = np.linspace(0, 1, 256)
        gradient = np.vstack((gradient, gradient))  # Make it 2D for imshow

        # Adjusting the plot to make space for the gradient
        plt.subplots_adjust(bottom=0.3)  # Increased bottom margin

        # Creating inset axes for the gradient
        axins = inset_axes(ax,
                           width="100%",  # Width as a percentage of the main axes
                           height="5%",   # Height as a percentage
                           loc='lower center',
                           bbox_to_anchor=(0, -0.25, 1, 1),  # Positioning below the main plot
                           bbox_transform=ax.transAxes,
                           borderpad=0)

        # Displaying the gradient
        axins.imshow(gradient, aspect='auto', cmap=cmap, extent=[0, 255, 0, 1], origin='lower')
        axins.axis('off')  # Hiding the axes

        # Adding Min and Max labels to the gradient
        axins.set_xticks([0, 255])
        axins.set_xticklabels(['Min', 'Max'], fontsize=axis_label_fontsize)

        # Saving the plot
        histogram_plot_path = os.path.join(output_dir, 'intensity_distribution_histogram.png')
        plt.savefig(histogram_plot_path, dpi=300, bbox_inches='tight')

        # Displaying the plot
        #plt.show()
        plt.close()
        #print("Intensity histogram displayed!")

    except Exception as e:
        print(f"Error in generate_intensity_histogram: {e}")

    # Additionally, create an Excel file with histogram data
    try:
        fluorescence_excel_path = os.path.join(output_dir, 'fluorescence-intensities.xlsx')
        histogram_df = df[['Value', 'Count']].copy()
        histogram_df.columns = ['values', 'count']  # Renaming columns

        # Rounding the 'values' column to 0 decimal places
        histogram_df['values'] = histogram_df['values'].round(0)

        # Creating the third column: V⋅c
        histogram_df['V⋅c'] = histogram_df['values'] * histogram_df['count']

        # Calculating the sum of V⋅c
        sum_vc = histogram_df['V⋅c'].sum()

        # Creating the fourth column: sum V⋅c with sum in the first data row
        histogram_df['sum V⋅c'] = np.nan  # Initialize with NaN
        if len(histogram_df) >= 1:
            histogram_df.at[0, 'sum V⋅c'] = sum_vc  # Assign sum to the first data row

        # Saving to Excel
        histogram_df.to_excel(fluorescence_excel_path, index=False)

    except Exception as e:
        print(f"Error in saving fluorescence intensities Excel: {e}")
        
def main():
    folder_path = 'C:/../'
    threshold_models = ["Default", "Huang", "Intermodes", "IsoData", "Li", "MaxEntropy", "Moments", "Otsu", "RenyiEntropy"] 

    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith('.lsm') and 'exp' in file_name:
            image_path = os.path.join(folder_path, file_name)
            output_dir = os.path.join(folder_path, os.path.splitext(file_name)[0].strip())  # Folder based on the original file name

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            print(f"Processing file: {file_name}")
            print("Thresholding the image …")
            # Run the function that performs analysis in ImageJ
            try:
                created_files = run_imagej_analysis(image_path, output_dir, threshold_models)
                print("✓ ImageJ thresholding analysis completed.")
            except FileNotFoundError as e:
                print(f"✗ ImageJ analysis failed: {e}")
                continue

            # Update output_dir variable if the folder name has changed
            new_output_dir = os.path.join(folder_path, shorten_file_name(os.path.basename(output_dir)))
            if output_dir != new_output_dir:
                if os.path.exists(new_output_dir):
                    # If a folder with the new name already exists, add a dash and number to get a unique name
                    counter = 2
                    temp_new_output_dir = f"{new_output_dir}-{counter}"
                    while os.path.exists(temp_new_output_dir):
                        counter += 1
                        temp_new_output_dir = f"{new_output_dir}-{counter}"
                    os.rename(output_dir, temp_new_output_dir)
                    output_dir = temp_new_output_dir
                else:
                    os.rename(output_dir, new_output_dir)
                    output_dir = new_output_dir
                #print(f"Output directory renamed to: {output_dir}")

            # Generate intensity histogram and save data to Excel
            print("Generating intensity histograms …")
            generate_intensity_histogram(output_dir)
            print("✓ Histograms generated.")

            # Prepare paths to CSV files for standard analysis
            output_csv_paths = {model: os.path.join(output_dir, f"results_{model}.csv") for model in threshold_models}

            # Prepare data for model training
            #print("Preparing data for model training...")
            X, y_categorical, labels = prepare_data(output_csv_paths)
            #print("Data prepared.")

            if X is None or y_categorical is None:
                print(f"No valid data found for {file_name}. Skipping...")
                continue

            # Calculate statistics and save for standard analysis
            stats_file_path = os.path.join(output_dir, 'statistics.xlsx')
            print("Calculating statistics for standard analysis...")
            model_stats = calculate_statistics_and_save(output_csv_paths, stats_file_path)

            # Calculate statistics and save for edge analysis
            edge_csv_paths = {model: os.path.join(output_dir, f"edges_{model}.csv") for model in threshold_models}
            edge_stats_file_path = os.path.join(output_dir, 'statistics_edges.xlsx')
            print("Calculating statistics for edge analysis...")
            edge_model_stats = calculate_edge_statistics_and_save(edge_csv_paths, edge_stats_file_path)

            # Generate comparison grid and save it
            generate_grid(
                output_dir,
                threshold_models,
                grid_size=5,  
                thumbnail_size=(200, 200),
                padding=30,
                font_size=40,
                frame_color=(0, 0, 0),
                text_color=(0, 0, 0),
                background_color=(255, 255, 255),
                frame_thickness=3,
                side_margin=150,
                top_margin=150,
                bottom_margin=150,
                text_padding=30,
                scale_factor=2
            )
            print("✓ Grid image saved.")

            # Merge results into Excel
            summary_excel_path = os.path.join(output_dir, 'summary_threshold_results.xlsx')
            merge_results_to_excel(
                file_name, 
                output_csv_paths, 
                threshold_models, 
                summary_excel_path,
                stats_file_path, 
                edge_stats_file_path
            )

            # Generate summary for edge analysis
            edge_summary_file_path = os.path.join(output_dir, 'summary_threshold_edges_results.xlsx')
            merge_results_to_excel(
                file_name, 
                edge_csv_paths, 
                threshold_models, 
                edge_summary_file_path,
                stats_file_path, 
                edge_stats_file_path
            )

            # Generate statistics plots and save
            plot_statistics(stats_file_path, edge_stats_file_path, os.path.join(output_dir, 'statistics_plots.png'))

            # Predict the best model based on statistical analysis
            best_models_within_1sd, models_within_2sd, models_within_3sd, model_stats_pred = predict_best_threshold_model(output_csv_paths)
            print("✓ Predictions of best threshold models saved.")

            # Save AI predictions to Excel
            save_predictions_to_excel(best_models_within_1sd, models_within_2sd, models_within_3sd, model_stats_pred, stats_file_path, os.path.join(output_dir, 'threshold_model_predictions.xlsx'))
            print("✓ Statistics and Shewhart predictions done.")

            # Move created files to appropriate folders
            # Create 'edges analysis' folder and move PNG files from edge analysis
            edges_analysis_folder = os.path.join(output_dir, 'edges analysis')
            if not os.path.exists(edges_analysis_folder):
                os.makedirs(edges_analysis_folder)

            edge_images = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.png') and 'edges_' in os.path.basename(f)]
            move_files_to_folder(edge_images, edges_analysis_folder)

            # Create 'edges csv' folder and move CSV files from edge analysis
            edges_csv_folder = os.path.join(output_dir, 'edges csv')
            if not os.path.exists(edges_csv_folder):
                os.makedirs(edges_csv_folder)

            edge_csv_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.csv') and 'edges_' in os.path.basename(f)]
            move_files_to_folder(edge_csv_files, edges_csv_folder)

            # Create 'macra' subfolder and move all macro files
            macra_folder = os.path.join(output_dir, 'macra')
            if not os.path.exists(macra_folder):
                os.makedirs(macra_folder)

            # Move macro files to 'macra' folder
            macro_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.ijm')]
            move_files_to_folder(macro_files, macra_folder)

            # Create 'csvs' subfolder and move model CSV files and 'intensity-histogram.csv'
            excel_results_folder = os.path.join(output_dir, 'csvs')
            if not os.path.exists(excel_results_folder):
                os.makedirs(excel_results_folder)

            # Move model result CSV files to 'csvs' folder
            csv_model_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.csv') and ('results_' in f or 'edges_' in f)]
            move_files_to_folder(csv_model_files, excel_results_folder)

            # Move 'intensity-histogram.csv' to 'csvs' folder
            intensity_histogram_csv = os.path.join(output_dir, 'intensity-histogram.csv')
            if os.path.exists(intensity_histogram_csv):
                move_files_to_folder([intensity_histogram_csv], excel_results_folder)
            
            shorten_file_names_in_folder(output_dir)
            print("✓ Memory cleaned.")
            print(f"✓ Processing for {file_name} completed.")
            
            gc.collect()


if __name__ == "__main__":
    main()
