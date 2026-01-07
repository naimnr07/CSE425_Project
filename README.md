Data Set: https://drive.google.com/file/d/16WayxwafeYu708qkvsjIGWuCzyodbnnj/view?usp=drivesdk


# CSE425 Project - VAE Music Clustering

Main File: cse425_project.py

Complete implementation of VAE-based music clustering for hybrid language tracks. All three tasks (Easy, Medium, Hard) in one file.

## Quick Start

1. Install Dependencies

Run this command to install required packages:

    pip install torch numpy pandas scikit-learn matplotlib seaborn tqdm umap-learn librosa

2. Run the Project

Automatic Mode (Runs all 3 tasks):

    python cse425_project.py

The script will automatically load the dataset from /content/dataset_positive_256_clean.pkl, use 1000 samples maximum, and run Easy, Medium, and Hard tasks one after another. All visualizations and metrics will be generated.

3. Manual Mode

Edit the file and set AUTO_RUN = False, then call specific tasks:

    run_easy_task('/path/to/dataset.pkl', is_pickle=True)
    run_medium_task('/path/to/dataset.pkl', is_pickle=True)
    run_hard_task('/path/to/dataset.pkl', is_pickle=True)

## Dataset Support

For Pickle Files (.pkl):

    run_easy_task('/content/dataset_positive_256_clean.pkl', is_pickle=True)

For CSV Files:

    run_easy_task('/path/to/lyrics/folder', is_pickle=False)

## Configuration

Key parameters you can edit in the file:

    DEFAULT_PKL_PATH = '/content/dataset_positive_256_clean.pkl'
    MAX_SAMPLES = 1000        (Limit dataset size)
    EPOCHS = 50               (Training epochs)
    BATCH_SIZE = 32           (Batch size)
    LATENT_DIM = 32           (Latent space dimensions)
    N_CLUSTERS = 5            (Number of clusters)

## What Each Task Does

Easy Task:
- Basic VAE architecture
- TF-IDF features from lyrics
- K-Means clustering
- PCA baseline comparison
- t-SNE and UMAP visualizations
- Metrics: Silhouette Score, Calinski-Harabasz Index

Medium Task:
- Enhanced VAE with 3 hidden layers
- Multiple clustering algorithms: K-Means, Agglomerative, DBSCAN
- Metrics: Silhouette, Calinski-Harabasz, Davies-Bouldin

Hard Task:
- Beta-VAE with beta value of 4.0
- Conditional VAE (CVAE)
- Autoencoder baseline
- Comprehensive comparison of all models
- Metrics: Silhouette, Calinski-Harabasz, Davies-Bouldin, ARI, NMI, Purity

## Output

Results are saved to the current directory:
- Training history plots
- t-SNE and UMAP visualizations
- Clustering metrics printed to console

## Troubleshooting

Error: No module named mymodel
The script handles this automatically with RestrictedUnpickler. If issues persist, run extract_pickle_data.py first.

Error: all elements of target should be between 0 and 1
This is fixed. Features are automatically normalized to the 0 to 1 range.

Out of Memory:
Reduce MAX_SAMPLES to 500, reduce BATCH_SIZE to 16, or reduce LATENT_DIM to 16.

DBSCAN produces no clusters:
This is normal if data density does not match parameters. Adjust eps and min_samples in the code if needed.

## For Google Colab

1. Upload cse425_project.py to Colab
2. Upload dataset to /content/dataset_positive_256_clean.pkl
3. Run the script:

    !python cse425_project.py


