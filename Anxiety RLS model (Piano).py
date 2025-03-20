#!/usr/bin/env python
# coding: utf-8

# # Library Utilized

# In[2]:


import pandas as pd
import numpy as np
from scipy.signal import butter, lfilter, gaussian, find_peaks
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis, entropy
from scipy.ndimage import filters
from numpy.fft import fft
from scipy.integrate import trapz
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import silhouette_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


# # Import and categorized RLS and Non-RLS datasets

# In[4]:


# Define file paths for the two categories
non_rls_files = ['PianoXRLS1.csv','PianoXRLS2.csv','PianoXRLS3.csv']
rls_files = ['PianoRLS1.csv', 'PianoRLS2.csv', 'PianoRLS3.csv']


# Dictionary to store datasets categorized by RLS and non-RLS
datasets = {'non_rls': {}, 'rls': {}}

# Loop to read and categorize non-RLS datasets
for i, file in enumerate(non_rls_files):
    datasets['non_rls'][f'non_rls_dataset_{i+1}'] = pd.read_csv(file)

# Loop to read and categorize RLS datasets
for i, file in enumerate(rls_files):
    datasets['rls'][f'rls_dataset_{i+1}'] = pd.read_csv(file)

# Display data types of each dataset for both categories
for category, category_datasets in datasets.items():
    print(f"Category: {category}")
    for name, dataset in category_datasets.items():
        print(f"Data types of {name}:")
        print(dataset.dtypes)
        print("\n")

# Access data types of individual datasets (corrected)
for category, category_datasets in datasets.items():
    for name, dataset in category_datasets.items():
        print(f"{name} Data Types: \n{dataset.dtypes}")


# # Low pass filter Non-RLS dataset

# In[5]:


# Define the butterworth low-pass filter
def butter_lowpass(cutoff, fs, order=4):
    nyquist = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyquist  # Normalize cutoff frequency
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

# Apply the filter to data
def lowpass_filter(data, cutoff, fs, order=4):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Segment the filtered signal
def segment_signal(data, window_size=128, overlap=64):
    segments = []
    for start in range(0, len(data) - window_size + 1, overlap):
        end = start + window_size
        segment = data[start:end]
        segments.append(segment)
    return segments

# Parameters for the filter
fs = 100       # Sampling frequency in Hz (adjust according to your data)
cutoff = 10     # Desired cutoff frequency of the filter in Hz
order = 5        # Order of the filter
window_size = 256  # Segment window size (adjust as needed)
overlap = 128     # Overlap between windows (adjust as needed)

# Create a DataFrame for segments instead of adding them to the original DataFrame
for i, file in enumerate(non_rls_files):
    datasets[f'dataset_{i+1}'] = pd.read_csv(file)
    
    # Extract columns from the current dataset
    dataset = datasets[f'dataset_{i+1}']
    time = dataset['time']
    gFz = dataset['gFz']
    
    # Apply the low-pass filter to gFz
    non_RLS_gFz_filtered = lowpass_filter(gFz, cutoff, fs, order)
    
    # Segment the filtered gFz data
    non_RLS_gFz_segments = segment_signal(non_RLS_gFz_filtered, window_size=window_size, overlap=overlap)
    
    # Convert segments to DataFrame where each row is a segment
    non_RLS_segments_df = pd.DataFrame(non_RLS_gFz_segments)
    
    # Store this DataFrame for further analysis or export
    datasets[f'dataset_{i+1}_segments'] = non_RLS_segments_df
    
   # Plot the original gFz signal in one graph
    plt.figure(figsize=(10, 6))
    plt.plot(time, gFz, label='Original gFz')
    plt.xlabel('Time [seconds]')
    plt.ylabel('gFz Amplitude')
    plt.title(f'Original non-RLS gFz Axis for Dataset {i+1}')
    plt.legend()
    plt.grid()
    plt.show()

    # Plot the filtered gFz signal in a separate graph
    plt.figure(figsize=(10, 6))
    plt.plot(time, non_RLS_gFz_filtered, label='Filtered gFz', color='orange')
    plt.xlabel('Time [seconds]')
    plt.ylabel('gFz Amplitude')
    plt.title(f'Low-pass Filtered (10 Hz cutoff) non_RLS gFz Axis for Dataset {i+1}')
    plt.legend()
    plt.grid()
    plt.show()

    # Display first few segments of the filtered data
    print(f"First two segments for dataset {i+1}:")
    print(non_RLS_segments_df.head(2))


# # Low pass filter RLS dataset

# In[6]:


# Define the butterworth low-pass filter
def butter_lowpass(cutoff, fs, order=4):
    nyquist = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyquist  # Normalize cutoff frequency
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

# Apply the filter to data
def lowpass_filter(data, cutoff, fs, order):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Segment the filtered signal
def segment_signal(data, window_size=128, overlap=64):
    segments = []
    for start in range(0, len(data) - window_size + 1, overlap):
        end = start + window_size
        segment = data[start:end]
        segments.append(segment)
    return segments

# Parameters for the filter
fs = 100       # Sampling frequency in Hz (adjust according to your data)
cutoff = 10     # Desired cutoff frequency of the filter in Hz
order = 5     # Order of the filter
window_size = 256  # Segment window size (adjust as needed)
overlap = 128     # Overlap between windows (adjust as needed)

# Create a DataFrame for segments instead of adding them to the original DataFrame
for i, file in enumerate(rls_files):
    datasets[f'dataset_{i+1}'] = pd.read_csv(file)
    
    # Extract columns from the current dataset
    dataset = datasets[f'dataset_{i+1}']
    time = dataset['time']
    gFz = dataset['gFz']
    
    # Apply the low-pass filter to gFz
    RLS_gFz_filtered = lowpass_filter(gFz, cutoff, fs, order)
    
    # Segment the filtered gFz data
    RLS_gFz_segments = segment_signal(RLS_gFz_filtered, window_size=window_size, overlap=overlap)
    
    # Convert segments to DataFrame where each row is a segment
    RLS_segments_df = pd.DataFrame(RLS_gFz_segments)
    
    # Store this DataFrame for further analysis or export
    datasets[f'dataset_{i+1}_segments'] = RLS_segments_df
    
   # Plot the original gFz signal in one graph
    plt.figure(figsize=(10, 6))
    plt.plot(time, gFz, label='Original gFz')
    plt.xlabel('Time [seconds]')
    plt.ylabel('gFz Amplitude')
    plt.title(f'Original RLS gFz Axis for Dataset {i+1}')
    plt.legend()
    plt.grid()
    plt.show()

    # Plot the filtered gFz signal in a separate graph
    plt.figure(figsize=(10, 6))
    plt.plot(time, RLS_gFz_filtered, label='Filtered gFz', color='orange')
    plt.xlabel('Time [seconds]')
    plt.ylabel('gFz Amplitude')
    plt.title(f'Low-pass Filtered (0.5 Hz cutoff) RLS gFz Axis for Dataset {i+1}')
    plt.legend()
    plt.grid()
    plt.show()

    # Display first few segments of the filtered data
    print(f"First two segments for dataset {i+1}:")
    print(RLS_segments_df.head(2))


# # Fast Fourier Transformation of filtered RLS and non-RLS datasets

# In[7]:


# Define the Butterworth low-pass filter
def butter_lowpass(cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

# Apply the low-pass filter
def lowpass_filter(data, cutoff, fs, order=4):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Plot FFT for a single dataset
def plot_fft(data, fs, title):
    data = data - np.mean(data)  # Normalize
    N = len(data)
    yf = fft(data)
    fft_magnitude = np.abs(yf[:N // 2])
    xf = np.fft.fftfreq(N, 1/fs)[:N//2]

    plt.figure(figsize=(10, 6))
    plt.plot(xf, fft_magnitude)
    plt.title(f'FFT for {title}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    if y_limit is not None:
        plt.ylim(0, y_limit)  # Set a uniform y-axis limit
    plt.show()

# Parameters
fs = 100       # Sampling frequency
cutoff = 10    # Cutoff frequency for the filter
order = 5      # Order of the filter
y_limit = 400  # Set the y-axis limit for comparability

# Plot FFT for each non-RLS dataset
for file in ['PianoXRLS1.csv','PianoXRLS2.csv','PianoXRLS3.csv']:
    df = pd.read_csv(file)
    gFz = df['gFz'].values
    gFz_filtered = lowpass_filter(gFz, cutoff, fs, order)
    plot_fft(gFz_filtered, fs, title=f'Non-RLS ({file})')

# Plot FFT for each RLS dataset
for file in ['PianoRLS1.csv', 'PianoRLS2.csv', 'PianoRLS3.csv']:
    df = pd.read_csv(file)
    gFz = df['gFz'].values
    gFz_filtered = lowpass_filter(gFz, cutoff, fs, order)
    plot_fft(gFz_filtered, fs, title=f'RLS ({file})')


# # Feature Extraction from RLS and non-RLS datasets

# In[8]:


# Define the Butterworth low-pass filter
def butter_lowpass(cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

# Apply the low-pass filter
def lowpass_filter(data, cutoff, fs, order=4):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Segment the filtered signal
def segment_signal(data, window_size=128, overlap=64):
    segments = []
    for start in range(0, len(data) - window_size + 1, overlap):
        segment = data[start:start + window_size]
        segments.append(segment)
    return segments

# Feature extraction function
def extract_features(segment, fs=100.0):
    features = {}
    segment = segment - np.mean(segment)  # Normalize segment

    # Time-domain features
    features['mean'] = np.mean(segment)
    features['std'] = np.std(segment)
    features['max'] = np.max(segment)
    features['min'] = np.min(segment)
    features['range'] = np.max(segment) - np.min(segment)
    features['rms'] = np.sqrt(np.mean(segment**2))
    features['median'] = np.median(segment)
    features['skewness'] = skew(segment)
    features['kurtosis'] = kurtosis(segment)
    features['variance'] = np.var(segment)
    features['energy'] = np.sum(np.square(segment))
    features['zero_crossings'] = ((segment[:-1] * segment[1:]) < 0).sum()
    features['angular_velocity'] = np.mean(np.gradient(segment))  # Approximate angular velocity
    features['signal_magnitude_area'] = np.sum(np.abs(segment))
    features['signal_vector_magnitude'] = np.sqrt(np.sum(segment**2))
    jerk_index = np.sum(np.abs(np.diff(segment)))
    features['jerk_index'] = jerk_index
    normalized_jerk_index = jerk_index / len(segment)
    features['normalized_jerk_index'] = normalized_jerk_index
    sway_path = np.sum(np.abs(np.diff(segment)))
    features['sway_path'] = sway_path
    mean_velocity = sway_path / len(segment)
    features['mean_velocity'] = mean_velocity
    sway_area = trapz(np.abs(segment), dx=1/fs)
    features['sway_area'] = sway_area
    
    # Find peaks and calculate largest distance between them
    peaks, _ = find_peaks(segment)
    if len(peaks) > 1:
        max_peak_distance = np.max(np.diff(peaks))
    else:
        max_peak_distance = 0  # No distance if there are fewer than 2 peaks
    features['max_peak_distance'] = max_peak_distance
    
    # Frequency-domain features
    N = len(segment)
    T = 1.0 / fs
    yf = fft(segment)
    xf = np.fft.fftfreq(N, T)[:N//2]
    psd = 2.0/N * np.abs(yf[:N//2])

    fft_values = np.fft.fft(segment)
    fft_magnitude = np.abs(fft_values)
    fft_power = np.square(fft_magnitude)
    
    features['dominant_freq'] = xf[np.argmax(psd)]
    features['psd_mean'] = np.mean(psd)
    features['psd_max'] = np.max(psd)
    features['coeff_sum'] = np.sum(np.abs(yf))
    psd_entropy = entropy(psd + 1e-12)  # Adding small value to avoid log(0)
    features['psd_entropy'] = psd_entropy
    non_zero_freqs = xf[psd > 0]  # Exclude zero frequency
    frequency_dispersion = np.max(non_zero_freqs) - np.min(non_zero_freqs)
    features['frequency_dispersion'] = frequency_dispersion
    centroidal_frequency = np.sum(xf * psd) / np.sum(psd)
    features['centroidal_frequency'] = centroidal_frequency
    cumulative_power = np.cumsum(psd)
    total_power = cumulative_power[-1]
    f50 = xf[np.where(cumulative_power >= 0.5 * total_power)][0]
    features['f50'] = f50
    f95 = xf[np.where(cumulative_power >= 0.95 * total_power)][0]
    features['f95'] = f95

    return features

# Define file paths for non-RLS and RLS files
non_rls_files = ['PianoXRLS1.csv','PianoXRLS2.csv','PianoXRLS3.csv']
rls_files = ['PianoRLS1.csv', 'PianoRLS2.csv', 'PianoRLS3.csv']

# Parameters
fs = 100       # Sampling frequency
cutoff = 10    # Cutoff frequency for the filter
order = 5      # Order of the filter
window_size = 128  # Segment window size
overlap = 64   # Overlap between segments

# Dictionary to store datasets and features
datasets = {'non_rls': [], 'rls': []}
features_list = []

# Load, filter, segment, and extract features for non-RLS data
for file in non_rls_files:
    df = pd.read_csv(file)
    gFz = df['gFz'].values
    gFz_filtered = lowpass_filter(gFz, cutoff, fs, order)
    segments = segment_signal(gFz_filtered, window_size, overlap)
    for segment in segments:
        features = extract_features(segment, fs=fs)
        features['label'] = 0  # Label non-RLS as 0
        features_list.append(features)

# Load, filter, segment, and extract features for RLS data
for file in rls_files:
    df = pd.read_csv(file)
    gFz = df['gFz'].values
    gFz_filtered = lowpass_filter(gFz, cutoff, fs, order)
    segments = segment_signal(gFz_filtered, window_size, overlap)
    for segment in segments:
        features = extract_features(segment, fs=fs)
        features['label'] = 1  # Label RLS as 1
        features_list.append(features)
        
# Create a DataFrame from the extracted features
features_df = pd.DataFrame(features_list)
features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
features_df.dropna(inplace=True)

# Display the first few rows of the DataFrame
print(features_df.head())


# # Summary statistics of features

# In[9]:


# Create a DataFrame from the features list
features_df = pd.DataFrame(features_list)

# Basic information about the dataset
print("Summary statistics for all features:\n")
print(features_df.describe())
print("\nCount of RLS and non-RLS segments:\n")
print(features_df['label'].value_counts())


# # Histogram plots of features

# In[10]:


# Histogram to check skewness
plt.figure(figsize=(15, 15))
num_features = len(features_df.columns) - 1  # Exclude the label column
rows = (num_features // 5) + 1
for i, feature in enumerate(features_df.columns[:-1], 1):  # Exclude the label column
    plt.subplot(8,4,i)
    sns.histplot(features_df[features_df['label'] == 0][feature], color='blue', label='Non-RLS', kde=True, bins=30, alpha=0.6)
    sns.histplot(features_df[features_df['label'] == 1][feature], color='orange', label='RLS', kde=True, bins=30, alpha=0.6)
    plt.title(f'Histogram of {feature} (RLS vs Non-RLS)')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.legend()

plt.tight_layout()
plt.show()


# # Boxplot of features

# In[11]:


# Boxplots to compare feature distributions for RLS vs. non-RLS
plt.figure(figsize=(20,20))
for i, feature in enumerate(features_df.columns[:-1], 1):  # Exclude the label column
    plt.subplot(8, 4, i)
    sns.boxplot(x='label', y=feature, data=features_df)
    plt.title(f'Distribution of {feature} by Label')
plt.tight_layout()
plt.show()


# # Plots after Box-cox transformation and Z-score normalization

# In[12]:


# Separate features and labels
X = features_df.drop(columns=['label'])
y = features_df['label']

# Ensure all features are positive for Box-Cox
X = X.copy()
for col in X.columns:
    X[col] = X[col] - X[col].min() + 1  # Shift to make all values positive

# 2. Box-Cox Transformation
box_cox_transformer = PowerTransformer(method='box-cox')  # Initialize Box-Cox transformer
X_box_cox = box_cox_transformer.fit_transform(X)  # Apply Box-Cox transformation

# 3. Z-score Normalization after Box-Cox Transformation
scaler = StandardScaler()
X_box_cox_zscore = scaler.fit_transform(X_box_cox)
features_df_box_cox_zscore = pd.DataFrame(X_box_cox_zscore, columns=X.columns)
features_df_box_cox_zscore['label'] = y

# Visualizing the distributions after Box-Cox Transformation and Z-score Normalization
plt.figure(figsize=(20, 20))
num_features = len(features_df_box_cox_zscore.columns[:-1])  # Number of features (exclude 'label')
rows = (num_features // 5) + 1

for i, feature in enumerate(features_df_box_cox_zscore.columns[:-1], 1):
    plt.subplot(9, 4, i)
    sns.boxplot(x='label', y=feature, data=features_df_box_cox_zscore, palette='Set2')
    plt.title(f'{feature}', fontsize=12)
    plt.xlabel('')
    plt.ylabel('Value', fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.suptitle('Z-score Normalized Distributions of Box-Cox Transformed Features', fontsize=16, y=1.02)
plt.tight_layout(pad=2.0)
plt.show()

plt.figure(figsize=(15, 15))
num_features = len(features_df_box_cox_zscore.columns) - 1  # Exclude the label column
rows = (num_features // 5) + 1
for i, feature in enumerate(features_df_box_cox_zscore[:-1], 1):  # Exclude the label column
    plt.subplot(9,4,i)
    sns.histplot(features_df_box_cox_zscore[features_df_box_cox_zscore['label'] == 0][feature], color='blue', label='Non-RLS', kde=True, bins=30, alpha=0.6)
    sns.histplot(features_df_box_cox_zscore[features_df_box_cox_zscore['label'] == 1][feature], color='orange', label='RLS', kde=True, bins=30, alpha=0.6)
    plt.title(f'Histogram of {feature} (RLS vs Non-RLS)')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.legend()

plt.tight_layout()
plt.show()



# # Principal Component Analysis

# In[13]:


# Specify the desired amount of variance to retain (e.g., 95%)
variance_threshold = 0.95

# Initialize PCA
pca = PCA(n_components=variance_threshold)

# Fit PCA on the normalized data (excluding the label column)
X_pca = pca.fit_transform(features_df_box_cox_zscore.drop(columns=['label']))

# Create a DataFrame for the PCA-transformed data
pca_columns = [f'PC{i+1}' for i in range(X_pca.shape[1])]
features_df_pca = pd.DataFrame(X_pca, columns=pca_columns)
features_df_pca['label'] = features_df_box_cox_zscore['label']

# Explained variance ratio of each principal component
explained_variance_ratio = pca.explained_variance_ratio_

# Plot the explained variance ratio
plt.figure(figsize=(10, 5))
plt.plot(np.cumsum(explained_variance_ratio))
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by Principal Components')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()

# Print the number of components chosen to capture the desired variance
print(f"Number of components chosen to retain {variance_threshold*100:.1f}% of variance: {pca.n_components_}")


# # Feature selection via random forest classifier

# In[14]:


# Ensure X and y are the same as those used for training
X = features_df_box_cox_zscore.drop(columns=['label'])
y = features_df_box_cox_zscore['label']

# Train-test split (if needed)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Initialize and train a Random Forest model
model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Feature importances
importances = model.feature_importances_

# Create a DataFrame for visualization (ensure the length matches)
feature_importances_df = pd.DataFrame({
    'Feature': X_train.columns,  # Use X_train columns for consistency
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(12, 8))
plt.barh(feature_importances_df['Feature'], feature_importances_df['Importance'], color='skyblue')
plt.xlabel('Importance Score')
plt.title('Feature Importances from Random Forest')
plt.gca().invert_yaxis()  # To show the highest importance on top
plt.grid(axis='x', linestyle='--', linewidth=0.5)
plt.show()

# Print the top N important features
print("Top Important Features:")
print(feature_importances_df.head(10))


# # Correlation Matrix

# In[15]:


# Correlation matrix to explore feature relationships
plt.figure(figsize=(20, 20))
sns.heatmap(features_df_box_cox_zscore.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix of Extracted Features")
plt.show()


# # Target Labelling via K-means clustering

# In[36]:


# Separate features and labels
X = features_df_box_cox_zscore[['max', 'normalized_jerk_index', 'sway_path', 'jerk_index', 'mean_velocity', 'zero_crossings','median', 'skewness','min','kurtosis']]
y = features_df_box_cox_zscore['label']

# Apply K-Means clustering
kmeans = KMeans(n_clusters=2, random_state=42)
features_df_box_cox_zscore['cluster'] = kmeans.fit_predict(features_df_box_cox_zscore.drop(columns=['label']))



# In[14]:


# Visualize clustering based on dominant frequency and number of peaks
plt.figure(figsize=(15, 9))
plt.scatter(features_df_box_cox_zscore['mean_velocity'], features_df_box_cox_zscore['max'], c=features_df_box_cox_zscore['cluster'], cmap='viridis')
plt.xlabel('Mean Velocity')
plt.ylabel('Max')
plt.title('Clustering Based on Frequency and Energy Features')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()


# In[37]:


pd.crosstab(features_df_box_cox_zscore['label'], features_df_box_cox_zscore['cluster'], rownames=['Label'], colnames=['Cluster'])


# In[27]:


# Invert cluster labels (swap 0 with 1 and vice versa)
features_df_box_cox_zscore['cluster'] = features_df_box_cox_zscore['cluster'].apply(lambda x: 1 if x == 0 else 0)

# Verify the inversion
inverted_cluster_counts = pd.crosstab(features_df_box_cox_zscore['label'], features_df_box_cox_zscore['cluster'])
print(inverted_cluster_counts)


# # Silhoutte Score

# In[38]:


# Check cluster purity with a contingency table
contingency_table = pd.crosstab(features_df_box_cox_zscore['label'], features_df_box_cox_zscore['cluster'])
print("Cluster Purity:\n", contingency_table)

# Evaluate clustering performance with silhouette score
sil_score = silhouette_score(features_df_box_cox_zscore.drop(columns=['label', 'cluster']), features_df_box_cox_zscore['cluster'])
print(f"Silhouette Score: {sil_score:.3f}")


# # Cluster distribution

# In[39]:


# Check distribution of clusters
class_distribution = features_df_box_cox_zscore['cluster'].value_counts()
print("Class distribution of clusters (count):")
print(class_distribution)

# Calculate percentage distribution of clusters
class_distribution_percentage = features_df_box_cox_zscore['cluster'].value_counts(normalize=True) * 100
print("\nClass distribution of clusters (percentage):")
print(class_distribution_percentage)


# # Classification via random undersampling and machine learning

# In[40]:


# Ensure X_subset has the selected features
X_subset = X  # Replace this with your selected feature DataFrame if needed

# Target variable
y = features_df_box_cox_zscore['cluster']

# Apply random undersampling to the entire dataset
undersampler = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = undersampler.fit_resample(X_subset, y)

# Split the undersampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Initialize the KNN classifier
knn = KNeighborsClassifier(n_neighbors=153)

# Train the KNN model on the training set
knn.fit(X_train, y_train)

# Predict on the training set for training metrics
y_train_pred = knn.predict(X_train)

# Predict on the test set
y_test_pred = knn.predict(X_test)

# Evaluate the model on the training set
train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred, average='weighted')
train_recall = recall_score(y_train, y_train_pred, average='weighted')
train_f1 = f1_score(y_train, y_train_pred, average='weighted')

# Print training evaluation metrics
print("Training Set Evaluation:")
print(f"Accuracy: {train_accuracy}")
print(f"Precision: {train_precision}")
print(f"Recall: {train_recall}")
print(f"F1 Score: {train_f1}")

# Compute and display the confusion matrix for the training set
train_conf_matrix = confusion_matrix(y_train, y_train_pred)
print("Training Confusion Matrix:")
print(train_conf_matrix)

# Evaluate the model on the test set
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred, average='weighted')
test_recall = recall_score(y_test, y_test_pred, average='weighted')
test_f1 = f1_score(y_test, y_test_pred, average='weighted')

# Print test evaluation metrics
print("\nTest Set Evaluation:")
print(f"Accuracy: {test_accuracy}")
print(f"Precision: {test_precision}")
print(f"Recall: {test_recall}")
print(f"F1 Score: {test_f1}")

# Compute and display the confusion matrix for the test set
test_conf_matrix = confusion_matrix(y_test, y_test_pred)
print("Test Confusion Matrix:")
print(test_conf_matrix)

# Print classification report for both training and test sets
train_class_report = classification_report(y_train, y_train_pred)
test_class_report = classification_report(y_test, y_test_pred)

print("\nTraining Set Classification Report:")
print(train_class_report)
print("\nTest Set Classification Report:")
print(test_class_report)


# In[41]:


# Ensure X_subset has the selected features
X_subset = X  # Replace this with your selected feature DataFrame if needed

# Target variable
y = features_df_box_cox_zscore['cluster']

# Apply random undersampling to the entire dataset
undersampler = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = undersampler.fit_resample(X_subset, y)

# Split the undersampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Initialize the SVM classifier
svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)

# Train the SVM model on the training set
svm.fit(X_train, y_train)

# Predict on the training set for training metrics
y_train_pred = svm.predict(X_train)

# Predict on the test set
y_test_pred = svm.predict(X_test)

# Evaluate the model on the training set
train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred, average='weighted')
train_recall = recall_score(y_train, y_train_pred, average='weighted')
train_f1 = f1_score(y_train, y_train_pred, average='weighted')

# Print training evaluation metrics
print("Training Set Evaluation:")
print(f"Accuracy: {train_accuracy}")
print(f"Precision: {train_precision}")
print(f"Recall: {train_recall}")
print(f"F1 Score: {train_f1}")

# Compute and display the confusion matrix for the training set
train_conf_matrix = confusion_matrix(y_train, y_train_pred)
print("Training Confusion Matrix:")
print(train_conf_matrix)

# Evaluate the model on the test set
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred, average='weighted')
test_recall = recall_score(y_test, y_test_pred, average='weighted')
test_f1 = f1_score(y_test, y_test_pred, average='weighted')

# Print test evaluation metrics
print("\nTest Set Evaluation:")
print(f"Accuracy: {test_accuracy}")
print(f"Precision: {test_precision}")
print(f"Recall: {test_recall}")
print(f"F1 Score: {test_f1}")

# Compute and display the confusion matrix for the test set
test_conf_matrix = confusion_matrix(y_test, y_test_pred)
print("Test Confusion Matrix:")
print(test_conf_matrix)

# Print classification report for both training and test sets
train_class_report = classification_report(y_train, y_train_pred)
test_class_report = classification_report(y_test, y_test_pred)

print("\nTraining Set Classification Report:")
print(train_class_report)
print("\nTest Set Classification Report:")
print(test_class_report)


# In[42]:


# Ensure X_subset has the selected features
X_subset = X  # Replace this with your selected feature DataFrame if needed

# Target variable
y = features_df_box_cox_zscore['cluster']

# Apply random undersampling to the entire dataset
undersampler = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = undersampler.fit_resample(X_subset, y)

# Split the undersampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Initialize the MLP classifier
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)

# Train the MLP model on the training set
mlp.fit(X_train, y_train)

# Predict on the training set for training metrics
y_train_pred = mlp.predict(X_train)

# Predict on the test set
y_test_pred = mlp.predict(X_test)

# Evaluate the model on the training set
train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred, average='weighted')
train_recall = recall_score(y_train, y_train_pred, average='weighted')
train_f1 = f1_score(y_train, y_train_pred, average='weighted')

# Print training evaluation metrics
print("Training Set Evaluation:")
print(f"Accuracy: {train_accuracy}")
print(f"Precision: {train_precision}")
print(f"Recall: {train_recall}")
print(f"F1 Score: {train_f1}")

# Compute and display the confusion matrix for the training set
train_conf_matrix = confusion_matrix(y_train, y_train_pred)
print("Training Confusion Matrix:")
print(train_conf_matrix)

# Evaluate the model on the test set
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred, average='weighted')
test_recall = recall_score(y_test, y_test_pred, average='weighted')
test_f1 = f1_score(y_test, y_test_pred, average='weighted')

# Print test evaluation metrics
print("\nTest Set Evaluation:")
print(f"Accuracy: {test_accuracy}")
print(f"Precision: {test_precision}")
print(f"Recall: {test_recall}")
print(f"F1 Score: {test_f1}")

# Compute and display the confusion matrix for the test set
test_conf_matrix = confusion_matrix(y_test, y_test_pred)
print("Test Confusion Matrix:")
print(test_conf_matrix)

# Print classification report for both training and test sets
train_class_report = classification_report(y_train, y_train_pred)
test_class_report = classification_report(y_test, y_test_pred)

print("\nTraining Set Classification Report:")
print(train_class_report)
print("\nTest Set Classification Report:")
print(test_class_report)


# In[43]:


from imblearn.under_sampling import RandomUnderSampler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Ensure X_subset has the selected features
X_subset = X  # Replace this with your selected feature DataFrame if needed

# Target variable
y = features_df_box_cox_zscore['cluster']

# Apply random undersampling to the entire dataset
undersampler = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = undersampler.fit_resample(X_subset, y)

# Split the undersampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Initialize the Naive Bayes model
nb_model = GaussianNB()

# Train the Naive Bayes model on the training set
nb_model.fit(X_train, y_train)

# Predict on the training set for training metrics
y_train_pred = nb_model.predict(X_train)

# Predict on the test set
y_test_pred = nb_model.predict(X_test)

# Evaluate the model on the training set
train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred, average='weighted')
train_recall = recall_score(y_train, y_train_pred, average='weighted')
train_f1 = f1_score(y_train, y_train_pred, average='weighted')

# Print training evaluation metrics
print("Training Set Evaluation:")
print(f"Accuracy: {train_accuracy}")
print(f"Precision: {train_precision}")
print(f"Recall: {train_recall}")
print(f"F1 Score: {train_f1}")

# Compute and display the confusion matrix for the training set
train_conf_matrix = confusion_matrix(y_train, y_train_pred)
print("Training Confusion Matrix:")
print(train_conf_matrix)

# Evaluate the model on the test set
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred, average='weighted')
test_recall = recall_score(y_test, y_test_pred, average='weighted')
test_f1 = f1_score(y_test, y_test_pred, average='weighted')

# Print test evaluation metrics
print("\nTest Set Evaluation:")
print(f"Accuracy: {test_accuracy}")
print(f"Precision: {test_precision}")
print(f"Recall: {test_recall}")
print(f"F1 Score: {test_f1}")

# Compute and display the confusion matrix for the test set
test_conf_matrix = confusion_matrix(y_test, y_test_pred)
print("Test Confusion Matrix:")
print(test_conf_matrix)

# Print classification report for both training and test sets
train_class_report = classification_report(y_train, y_train_pred)
test_class_report = classification_report(y_test, y_test_pred)

print("\nTraining Set Classification Report:")
print(train_class_report)
print("\nTest Set Classification Report:")
print(test_class_report)


# # Evaluation Metrics Plot

# In[3]:


# Define data
algorithms = ['NB', 'KNN', 'SVM', 'MLP']
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

training_scores = {
    'Accuracy': [93.68, 92.79, 99.26, 99.56],
    'Precision': [93.81, 92.80, 99.27, 99.56],
    'Recall': [93.68, 92.79, 99.26, 99.56],
    'F1-Score': [93.68, 92.79, 99.26, 99.56]
}

test_scores = {
    'Accuracy': [92.47, 90.75, 98.29, 98.63],
    'Precision': [92.67, 90.77, 98.30, 98.67],
    'Recall': [92.47, 90.75, 98.29, 98.63],
    'F1-Score': [92.46, 90.75, 98.29, 98.63]
}

# Plot bar chart comparison for training vs test for each metric
def plot_training_vs_test(training_scores, test_scores, algorithms, metrics):
    x = np.arange(len(algorithms))  # the label locations
    width = 0.35  # the width of the bars

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(10, 6))

        bars1 = ax.bar(x - width/2, training_scores[metric], width, label='Training')
        bars2 = ax.bar(x + width/2, test_scores[metric], width, label='Test')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_xlabel('Algorithms')
        ax.set_ylabel(f'{metric} (%)')
        ax.set_title(f'Training vs Test Comparison: {metric}')
        ax.set_xticks(x)
        ax.set_xticklabels(algorithms)
        ax.legend()

        # Add value annotations
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

        plt.ylim(90, 100)  # Set y-axis limit
        plt.tight_layout()
        plt.show()

# Call the function to plot
plot_training_vs_test(training_scores, test_scores, algorithms, metrics)

