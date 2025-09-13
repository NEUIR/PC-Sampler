import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import seaborn as sns
import numpy as np
import json, os
import argparse

def load_json_or_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        if file_path.endswith('.jsonl'):
            for line in f:
                data.append(json.loads(line))
        elif file_path.endswith('.json'):
            data = json.load(f)
        else:
            raise ValueError("Unsupported file format. Please use .json or .jsonl")
    return data

def _plot_heatmap_to_file(data, vmin, vmax, filename):
    figsize = (6.8, 5.5)
    fig = plt.figure(figsize=figsize)
    right_margin = 0.82
    gs = gridspec.GridSpec(1, 1, left=0.15, right=right_margin, top=0.98, bottom=0.15)
    ax = fig.add_subplot(gs[0])
    hmap = sns.heatmap(
        data,
        cmap='YlGnBu',
        annot=False,
        cbar=False,
        square=True,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        rasterized=True
    )
    matrix_size = data.shape[0]
    ticks = np.linspace(0, matrix_size, 5, dtype=int)
    ax.set_yticks(ticks)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks, fontsize=24, rotation=0)
    ax.set_ylabel('Steps', fontsize=28)
    ax.set_yticklabels(ticks, fontsize=24)
    ax.tick_params(axis='y', which='both', left=True, right=False)
    ax.set_title("") 
    ax_pos = ax.get_position()
    shrink_factor = 0.9 
    cax_width = 0.02   
    cax_pad = 0.01
    cax_left = ax_pos.x1 + cax_pad
    cax_height = ax_pos.height * shrink_factor
    cax_bottom = ax_pos.y0 + (ax_pos.height * (1 - shrink_factor) / 2.0)
    cax = fig.add_axes([cax_left, cax_bottom, cax_width, cax_height])
    cbar = fig.colorbar(hmap.collections[0], cax=cax)
    for spine in cbar.ax.spines.values():
        spine.set_visible(False)
    cbar.set_ticks([vmin, vmax])
    cbar.set_ticklabels(['0', '1'])
    cbar.ax.yaxis.set_ticks_position('right')
    cbar.ax.yaxis.set_label_position('right')
    cbar.ax.tick_params(axis='y', pad=5, labelsize=24)
    cbar.set_label('Decoding Order', size=28, labelpad=0)
    plt.savefig(filename, format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.0)
    plt.close(fig)
    print(f"Saved figure: {filename}")

def _plot_barchart_to_file(bar_data, filename, y_max=None):
    fig = plt.figure(figsize=(6.4, 5.5))
    gs = gridspec.GridSpec(1, 1, left=0.18, right=0.95, top=0.92, bottom=0.15)
    ax_bar = fig.add_subplot(gs[0])
    bar_labels = ['L-to-R', 'Semi-AR', 'Conf.', 'Entropy', 'Margin']
    bar_fills = ['#1a4b8c', '#3292dc', '#4cc3d9', '#76c2af', '#a5d992']
    bar_edges = ['#0f3a6b', '#1a62a0', '#2d8b9e', '#3d7d6a', '#6aa35b']
    x = np.arange(len(bar_labels))
    width = 0.6 
    bar_container = ax_bar.bar(x, bar_data, width, 
                                  color=bar_fills,
                                  edgecolor=bar_edges,
                                  linewidth=2)
    if y_max is not None:
        ax_bar.set_ylim(top=y_max)
    else:
        ax_bar.set_ylim(top=max(bar_data) * 2.1)
    for bar in bar_container:
        height = bar.get_height()
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2.,
            height + 1.5,
            f'{height:.1f}',
            ha='center', va='bottom',
            fontsize=22
        )
    ax_bar.set_title("")
    ax_bar.set_ylabel('Accuracy (%)', fontsize=28)
    ax_bar.tick_params(axis='y', labelleft=True, left=True, labelsize=24)
    ax_bar.set_xticks([])
    ax_bar.set_xticklabels([])
    legend_handles = [Patch(facecolor=bar_fills[i], edgecolor=bar_edges[i], label=bar_labels[i]) for i in range(len(bar_labels))]
    ax_bar.legend(handles=legend_handles, 
                  loc='upper right', 
                  fontsize=22,
                  frameon=False, 
                  borderaxespad=0.0,)
    ax_bar.grid(axis='y', linestyle='--', linewidth=1.2, color='#c2c2c2', alpha=0.6, dashes=(5, 7))
    ax_bar.set_axisbelow(True)
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    ax_bar.spines['left'].set_visible(True) 
    plt.savefig(filename, format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.0)
    plt.close(fig)
    print(f"Saved figure: {filename}")

def generate_individual_figures(
    confidence_result: np.ndarray,
    entropy_result: np.ndarray,
    margin_result: np.ndarray,
    bar_chart_data: list,
    basename: str = 'figure_1'
):
    try:
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
        plt.rcParams['mathtext.fontset'] = 'cm'
    except:
        print("Times New Roman font not found. Using default font.")
    heatmap_datas = [confidence_result, entropy_result, margin_result]
    vmin = min(np.min(d) for d in heatmap_datas)
    vmax = max(np.max(d) for d in heatmap_datas)
    _plot_heatmap_to_file(
        data=heatmap_datas[0], vmin=vmin, vmax=vmax,
        filename=f"{basename}_1.pdf"
    )
    _plot_heatmap_to_file(
        data=heatmap_datas[1], vmin=vmin, vmax=vmax,
        filename=f"{basename}_2.pdf"
    )
    _plot_heatmap_to_file(
        data=heatmap_datas[2], vmin=vmin, vmax=vmax,
        filename=f"{basename}_3.pdf"
    )
    _plot_barchart_to_file(
        bar_data=bar_chart_data,
        filename=f"{basename}_4.pdf"
    )

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='humaneval')
    parser.add_argument('--data_path', type=str, default='./heatmap_params/humaneval.json')
    args = parser.parse_args()
    
    data_path = args.data_path
    task = args.task
    data = load_json_or_jsonl(data_path)
    confidence_result = np.array(data['confidence_result'])
    entropy_result = np.array(data['entropy_result'])
    margin_result = np.array(data['margin_result'])
    if confidence_result.ndim == 3:
        confidence_result = confidence_result.squeeze()
    if entropy_result.ndim == 3:
        entropy_result = entropy_result.squeeze()
    if margin_result.ndim == 3:
        margin_result = margin_result.squeeze()
    results = {'humaneval':[44.51, 39.02, 8.54, 3.05, 13.41], 
               'mbpp':[45.20, 45.20, 33.96, 28.57, 36.30],
               'math500':[32.80, 27.60, 3.4, 3.8, 1.8],
               'countdown':[36.30, 32.60, 34.0, 33.8, 33.9],
               'sudoku':[0.0, 0.0, 23.80, 1.6, 26.6],
               'gsm8k':[78.24, 77.86, 6.75, 2.2, 11.07],
               'gpqa':[27.90, 27.68, 27.90, 28.35, 28.35]}
    
    file_path = f'heatmap_results/{task}'
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    generate_individual_figures(
        confidence_result=confidence_result,
        entropy_result=entropy_result,
        margin_result=margin_result,
        bar_chart_data=results[task],
        basename=f"{file_path}/{task}"
    )