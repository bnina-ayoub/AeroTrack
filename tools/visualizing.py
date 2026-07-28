import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_prepare_routing_data(file_path: str) -> pd.DataFrame:
    dataframe = pd.read_csv(file_path)
    dataframe['deep_pipeline_frames'] = dataframe['total_frames'] - dataframe['early_exits']
    return dataframe.sort_values('early_exit_ratio', ascending=False).reset_index(drop=True)

def apply_publication_aesthetics() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': 12})

def render_routing_distribution_chart(data: pd.DataFrame, output_filename: str) -> None:
    figure, axes = plt.subplots(figsize=(14, 8))
    
    axes.bar(
        data['sequence'], 
        data['deep_pipeline_frames'], 
        color='#1f77b4', 
        label='Deep Pipeline (P4/P5)'
    )
    axes.bar(
        data['sequence'], 
        data['early_exits'], 
        bottom=data['deep_pipeline_frames'], 
        color='#ff7f0e', 
        label='Early Exit (P3)'
    )
    
    axes.set_title('Frame Routing Distribution per Sequence', fontsize=16, fontweight='bold')
    axes.set_ylabel('Number of Frames')
    axes.set_xlabel('Sequence')
    axes.tick_params(axis='x', rotation=90, labelsize=10)
    axes.legend()
    
    figure.tight_layout()
    figure.savefig(output_filename, dpi=300)
    plt.close(figure)

def generate_analytical_assets(csv_filepath: str) -> None:
    apply_publication_aesthetics()
    routing_data = load_and_prepare_routing_data(csv_filepath)
    render_routing_distribution_chart(routing_data, 'frame_routing_stacked.png')

if __name__ == "__main__":
    generate_analytical_assets('/home/bninaos/UAV-Tracking-Project/AeroTrack/YOLOX_outputs/UAVSwarm Output/aerotrack_early_exit_nwd/early_exit_stats.csv')