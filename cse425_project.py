
# IMPORTS
import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score
)
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("UMAP not available. Install with: pip install umap-learn")

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("Librosa not available. Audio features will be disabled.")

try:
    import kagglehub
    KAGGLEHUB_AVAILABLE = True
except ImportError:
    KAGGLEHUB_AVAILABLE = False
    print("Kagglehub not available. Install with: pip install kagglehub")


# ============================================================================
# DATASET LOADING - Updated for .pkl support
# ============================================================================
class LyricsDataset(Dataset):
    """Dataset class for lyrics data - supports CSV and .pkl formats."""
    
    def __init__(self, data_path, max_features=5000, normalize=True, is_pickle=False, max_samples=1000):
        
        self.data_path = data_path
        self.max_features = max_features
        self.normalize = normalize
        self.is_pickle = is_pickle or data_path.endswith('.pkl')
        self.max_samples = max_samples
        self.scaler = StandardScaler() if normalize else None
        
        # Load data based on format
        if self.is_pickle:
            self.data, self.labels, self.features = self._load_pickle()
        else:
            self.data, self.labels = self._load_data()
            self.features = self._extract_features()
        
        # If we have text data but no features, extract TF-IDF features
        if self.data is not None and self.features is None:
            print("Text data found, extracting TF-IDF features...")
            self.features = self._extract_features()
        
        # Verify we have features
        if self.features is None:
            raise ValueError("No features extracted. Check dataset format and extraction logic.")
        
        # Normalize features to [0, 1] range for binary cross-entropy loss
        # TF-IDF values are non-negative, so we just need to scale to [0, 1]
        if self.features.max() > 1.0 or self.features.min() < 0.0:
            print("Normalizing features to [0, 1] range for VAE training...")
            feature_min = self.features.min()
            feature_max = self.features.max()
            if feature_max > feature_min:
                self.features = (self.features - feature_min) / (feature_max - feature_min)
            else:
                self.features = np.zeros_like(self.features)
        
        # Limit to max_samples
        if len(self.features) > self.max_samples:
            print(f"Limiting dataset to {self.max_samples} samples (from {len(self.features)})")
            np.random.seed(42)  # For reproducibility
            indices = np.random.choice(len(self.features), self.max_samples, replace=False)
            indices = np.sort(indices)  # Keep original order
            self.features = self.features[indices]
            if self.labels is not None:
                if isinstance(self.labels, (list, tuple)):
                    self.labels = [self.labels[i] for i in indices]
                elif isinstance(self.labels, np.ndarray):
                    self.labels = self.labels[indices]
                else:
                    self.labels = [self.labels[i] for i in indices]
            if self.data is not None:
                if isinstance(self.data, (list, tuple)):
                    self.data = [self.data[i] for i in indices]
                elif isinstance(self.data, np.ndarray):
                    self.data = self.data[indices]
                else:
                    self.data = [self.data[i] for i in indices]
        
        if normalize and self.features is not None:
            # For binary cross-entropy loss, we need [0, 1] range
            # So we'll do min-max normalization instead of StandardScaler
            feature_min = self.features.min()
            feature_max = self.features.max()
            if feature_max > feature_min:
                self.features = (self.features - feature_min) / (feature_max - feature_min)
            else:
                self.features = np.zeros_like(self.features)
            print(f"Features normalized to [0, 1] range (min: {self.features.min():.4f}, max: {self.features.max():.4f})")
    
    def _load_pickle(self):
        """Load dataset from .pkl file."""
        print(f"Loading dataset from pickle file: {self.data_path}")
        
        try:
            # Try standard pickle load
            with open(self.data_path, 'rb') as f:
                data = pickle.load(f)
        except (ModuleNotFoundError, AttributeError) as e:
            # If there's a missing module dependency, try alternative loading
            print(f"Warning: {str(e)}")
            print("Attempting alternative loading method...")
            
            try:
                # Try loading with restricted unpickling
                class RestrictedUnpickler(pickle.Unpickler):
                    def find_class(self, module, name):
                        # Allow only safe modules, replace missing ones with dummy classes
                        if 'mymodel' in module or '__main__' in module:
                            return type(name, (), {})  # Return dummy class
                        # Allow standard modules
                        allowed_modules = ['numpy', 'pandas', 'builtins', '__builtin__']
                        if any(module.startswith(m) for m in allowed_modules):
                            return super().find_class(module, name)
                        # For other missing modules, return dummy
                        return type(name, (), {})
                
                with open(self.data_path, 'rb') as f:
                    unpickler = RestrictedUnpickler(f)
                    data = unpickler.load()
            except Exception as e2:
                print(f"Alternative loading failed: {str(e2)}")
                print("Trying to extract data directly from file...")
                
                # Last resort: try to read the file as if it contains raw arrays
                try:
                    # Try to extract using joblib if available
                    try:
                        import joblib
                        with open(self.data_path, 'rb') as f:
                            data = joblib.load(f)
                    except:
                        # If joblib also fails, try to manually extract numpy arrays
                        print("Attempting to extract numpy arrays directly...")
                        raise e  # Re-raise original error with instructions
                except Exception as e3:
                    print(f"\nERROR: Cannot load pickle file due to missing dependencies.")
                    print(f"Original error: {str(e)}")
                    print(f"\nSOLUTION OPTIONS:")
                    print("1. Install missing module: pip install mymodel")
                    print("2. Extract data from pickle file manually")
                    print("3. Re-save the pickle file without custom classes")
                    print("\nTrying to provide manual extraction instructions...")
                    raise e
            
            # Handle different pickle formats
            if isinstance(data, dict):
                # Dictionary format: {'features': ..., 'labels': ..., etc.}
                print("Detected dictionary format")
                features = None
                labels = None
                text_data = None
                
                # Try common key names
                for key in data.keys():
                    key_lower = key.lower()
                    if 'feature' in key_lower or 'data' in key_lower or 'x' in key_lower:
                        features = data[key]
                        print(f"Found features in key '{key}': shape {features.shape if hasattr(features, 'shape') else len(features)}")
                    if 'label' in key_lower or 'y' in key_lower or 'target' in key_lower:
                        labels = data[key]
                        print(f"Found labels in key '{key}': length {len(labels) if hasattr(labels, '__len__') else 'N/A'}")
                    if 'text' in key_lower or 'lyric' in key_lower or 'song' in key_lower:
                        text_data = data[key]
                        print(f"Found text data in key '{key}'")
                
                # If features are not pre-extracted, try to extract from text
                if features is None and text_data is not None:
                    print("No pre-extracted features found, extracting from text data...")
                    if isinstance(text_data, list):
                        self.data = text_data
                    elif isinstance(text_data, np.ndarray):
                        self.data = text_data.tolist()
                    else:
                        self.data = [str(x) for x in text_data]
                    return self.data, labels, None  # Will extract features later
                
                # Convert features to numpy array if needed
                if features is not None:
                    if isinstance(features, (list, tuple)):
                        features = np.array(features)
                    elif isinstance(features, pd.DataFrame):
                        features = features.values
                    
                    # Handle labels
                    if labels is None:
                        labels = [None] * len(features)
                    elif isinstance(labels, (list, tuple, np.ndarray)):
                        labels = list(labels)
                    else:
                        labels = [labels] * len(features)
                    
                    print(f"Loaded {len(features)} samples with {features.shape[1] if len(features.shape) > 1 else 'scalar'} features")
                    return None, labels, features  # Return features directly
                
            elif isinstance(data, (list, tuple)):
                # List/tuple format
                print("Detected list/tuple format")
                if len(data) == 0:
                    raise ValueError("Empty data list")
                
                first_item = data[0]
                
                if isinstance(first_item, str):
                    # List of text strings
                    self.data = list(data)
                    return self.data, [None] * len(self.data), None
                elif isinstance(first_item, np.ndarray) or isinstance(first_item, (list, tuple)):
                    # List of feature vectors
                    try:
                        features = np.array(data)
                        return None, [None] * len(features), features
                    except:
                        # If direct conversion fails, try to extract
                        features = []
                        for item in data:
                            if isinstance(item, np.ndarray):
                                features.append(item.flatten())
                            elif isinstance(item, (list, tuple)):
                                features.append(np.array(item).flatten())
                            else:
                                # Object - try to extract numeric attributes
                                attrs = [getattr(item, attr) for attr in dir(item) 
                                        if not attr.startswith('_') and 
                                        isinstance(getattr(item, attr), (int, float, np.ndarray))]
                                if attrs:
                                    features.append(np.array(attrs).flatten())
                                else:
                                    raise ValueError(f"Cannot extract features from object: {type(item)}")
                        features = np.array(features)
                        return None, [None] * len(features), features
                else:
                    # List of objects - try to extract numeric features
                    print(f"Detected objects of type: {type(first_item)}")
                    print("Attempting to extract numeric features from objects...")
                    
                    features = []
                    labels = []
                    
                    # First, inspect one object to understand structure
                    sample_item = data[0]
                    print(f"Inspecting object attributes...")
                    
                    # Get all attributes
                    all_attrs = dir(sample_item)
                    print(f"Available attributes: {[a for a in all_attrs if not a.startswith('__')][:20]}...")
                    
                    # Try to get __dict__ if available
                    item_dict = {}
                    if hasattr(sample_item, '__dict__'):
                        item_dict = sample_item.__dict__
                        print(f"Object __dict__ keys: {list(item_dict.keys())[:20]}...")
                    
                    text_data_found = False
                    text_attrs = []
                    numeric_attrs_list = []
                    
                    for item in data[:10]:  # Check first 10 items
                        # Try common attribute names for features/embeddings
                        item_features = None
                        item_text = None
                        
                        # Priority 1: Look for embedding/feature attributes
                        for attr_name in ['embedding', 'features', 'feature', 'vector', 'vec', 'data', 'x', 'X', 
                                        'encoded', 'representation', 'latent', 'hidden']:
                            try:
                                if hasattr(item, attr_name):
                                    val = getattr(item, attr_name)
                                    if isinstance(val, np.ndarray) and len(val.shape) > 0:
                                        item_features = val.flatten()
                                        break
                                    elif isinstance(val, (list, tuple)) and len(val) > 0:
                                        item_features = np.array(val).flatten()
                                        break
                            except:
                                pass
                        
                        # Priority 2: Look for text data (we'll extract TF-IDF)
                        if item_features is None:
                            for attr_name in ['comment', 'text', 'lyrics', 'lyric', 'song', 'content', 
                                            'message', 'description', 'title']:
                                try:
                                    if hasattr(item, attr_name):
                                        val = getattr(item, attr_name)
                                        if isinstance(val, str) and len(val) > 0:
                                            item_text = val
                                            text_data_found = True
                                            break
                                except:
                                    pass
                        
                        # Priority 3: Try __dict__ approach
                        if item_features is None and item_text is None:
                            try:
                                if hasattr(item, '__dict__'):
                                    d = item.__dict__
                                    # Look for numpy arrays or lists
                                    for key, val in d.items():
                                        if isinstance(val, np.ndarray) and len(val.shape) > 0:
                                            item_features = val.flatten()
                                            break
                                        elif isinstance(val, (list, tuple)) and len(val) > 0 and isinstance(val[0], (int, float)):
                                            item_features = np.array(val).flatten()
                                            break
                                        elif isinstance(val, str) and len(val) > 10:
                                            item_text = val
                                            text_data_found = True
                                            break
                            except:
                                pass
                        
                        # Priority 4: Extract all numeric values from __dict__
                        if item_features is None and item_text is None:
                            try:
                                if hasattr(item, '__dict__'):
                                    d = item.__dict__
                                    numeric_values = []
                                    for key, val in d.items():
                                        if isinstance(val, (int, float)):
                                            numeric_values.append(float(val))
                                        elif isinstance(val, np.ndarray):
                                            numeric_values.extend(val.flatten().tolist())
                                        elif isinstance(val, (list, tuple)):
                                            for v in val:
                                                if isinstance(v, (int, float)):
                                                    numeric_values.append(float(v))
                                    if numeric_values:
                                        item_features = np.array(numeric_values)
                            except:
                                pass
                        
                        # Store findings
                        if item_features is not None:
                            numeric_attrs_list.append(item_features)
                        if item_text is not None:
                            text_attrs.append(item_text)
                    
                    # Now process all items based on what we found
                    print(f"Text data found: {text_data_found}, Numeric features found: {len(numeric_attrs_list) > 0}")
                    
                    if text_data_found:
                        # Extract text from all items and use TF-IDF
                        print("Extracting text data from objects for TF-IDF feature extraction...")
                        text_data = []
                        labels = []
                        
                        for item in data:
                            item_text = None
                            item_label = None
                            
                            # Try to get text
                            for attr_name in ['comment', 'text', 'lyrics', 'lyric', 'song', 'content']:
                                try:
                                    if hasattr(item, attr_name):
                                        val = getattr(item, attr_name)
                                        if isinstance(val, str):
                                            item_text = val
                                            break
                                except:
                                    pass
                            
                            # Try __dict__
                            if item_text is None and hasattr(item, '__dict__'):
                                d = item.__dict__
                                for key, val in d.items():
                                    if isinstance(val, str) and len(val) > 10:
                                        item_text = val
                                        break
                            
                            # Try to get label
                            for label_name in ['label', 'y', 'target', 'class', 'category', 'sentiment']:
                                try:
                                    if hasattr(item, label_name):
                                        val = getattr(item, label_name)
                                        if isinstance(val, (int, float, str)):
                                            item_label = val
                                            break
                                except:
                                    pass
                            
                            if item_text:
                                text_data.append(item_text)
                                labels.append(item_label)
                        
                        if text_data:
                            print(f"Extracted {len(text_data)} text samples")
                            self.data = text_data
                            return self.data, labels, None  # Will extract TF-IDF features later
                    
                    if numeric_attrs_list:
                        # Extract numeric features from all items
                        print("Extracting numeric features from objects...")
                        features = []
                        labels = []
                        
                        for item in data:
                            item_features = None
                            item_label = None
                            
                            # Try embedding/feature attributes
                            for attr_name in ['embedding', 'features', 'feature', 'vector', 'vec', 'data']:
                                try:
                                    if hasattr(item, attr_name):
                                        val = getattr(item, attr_name)
                                        if isinstance(val, np.ndarray) and len(val.shape) > 0:
                                            item_features = val.flatten()
                                            break
                                        elif isinstance(val, (list, tuple)) and len(val) > 0:
                                            item_features = np.array(val).flatten()
                                            break
                                except:
                                    pass
                            
                            # Try __dict__
                            if item_features is None and hasattr(item, '__dict__'):
                                d = item.__dict__
                                for key, val in d.items():
                                    if isinstance(val, np.ndarray) and len(val.shape) > 0:
                                        item_features = val.flatten()
                                        break
                                    elif isinstance(val, (list, tuple)) and len(val) > 0:
                                        try:
                                            arr = np.array(val)
                                            if arr.dtype in [np.int32, np.int64, np.float32, np.float64]:
                                                item_features = arr.flatten()
                                                break
                                        except:
                                            pass
                            
                            # Extract all numeric values
                            if item_features is None and hasattr(item, '__dict__'):
                                d = item.__dict__
                                numeric_values = []
                                for key, val in d.items():
                                    if isinstance(val, (int, float)):
                                        numeric_values.append(float(val))
                                    elif isinstance(val, np.ndarray):
                                        numeric_values.extend(val.flatten().tolist())
                                if numeric_values:
                                    item_features = np.array(numeric_values)
                            
                            # Try to get label
                            for label_name in ['label', 'y', 'target', 'class', 'category']:
                                try:
                                    if hasattr(item, label_name):
                                        val = getattr(item, label_name)
                                        if isinstance(val, (int, float, str)):
                                            item_label = val
                                            break
                                except:
                                    pass
                            
                            if item_features is not None and len(item_features) > 0:
                                features.append(item_features)
                                labels.append(item_label)
                        
                        if features:
                            # Pad to same length
                            max_len = max(len(f) for f in features)
                            padded_features = []
                            for f in features:
                                if len(f) < max_len:
                                    f = np.pad(f, (0, max_len - len(f)), 'constant', constant_values=0)
                                padded_features.append(f)
                            features = np.array(padded_features, dtype=np.float32)
                            print(f"Extracted {len(features)} samples with {features.shape[1]} features")
                            return None, labels if any(l is not None for l in labels) else [None] * len(features), features
                    
                    if features:
                        # Convert to numpy array
                        # Find max length to pad if needed
                        max_len = max(len(f) for f in features)
                        if max_len > 0:
                            # Pad shorter feature vectors with zeros
                            padded_features = []
                            for f in features:
                                if len(f) < max_len:
                                    f = list(f) + [0.0] * (max_len - len(f))
                                padded_features.append(f)
                            features = np.array(padded_features, dtype=np.float32)
                            
                            # Ensure all features have same length
                            if len(features) > 0:
                                print(f"Extracted {len(features)} samples with {features.shape[1]} features")
                                return None, labels if any(l is not None for l in labels) else [None] * len(features), features
                    
                    # If all else fails, try to use the object itself (last resort)
                    raise ValueError(f"Cannot extract numeric features from objects of type: {type(first_item)}")
                    
            elif isinstance(data, np.ndarray):
                # Numpy array format
                print(f"Detected numpy array format: shape {data.shape}")
                if len(data.shape) == 1:
                    # 1D array - might be labels or text indices
                    features = None
                    labels = data
                    return None, labels.tolist(), None
                elif len(data.shape) == 2:
                    # 2D array - feature matrix
                    return None, [None] * data.shape[0], data
                else:
                    # Higher dimensional - flatten or reshape
                    print(f"Reshaping array from {data.shape} to 2D")
                    features = data.reshape(data.shape[0], -1)
                    return None, [None] * features.shape[0], features
                    
            elif isinstance(data, pd.DataFrame):
                # Pandas DataFrame format
                print("Detected pandas DataFrame format")
                # Try to identify columns
                text_col = None
                label_col = None
                feature_cols = []
                
                for col in data.columns:
                    col_lower = col.lower()
                    if 'lyric' in col_lower or 'text' in col_lower or 'song' in col_lower:
                        text_col = col
                    elif 'label' in col_lower or 'lang' in col_lower or 'y' in col_lower:
                        label_col = col
                    else:
                        # Assume numeric columns are features
                        if pd.api.types.is_numeric_dtype(data[col]):
                            feature_cols.append(col)
                
                if text_col:
                    self.data = data[text_col].astype(str).tolist()
                    labels = data[label_col].tolist() if label_col else [None] * len(data)
                    return self.data, labels, None
                elif feature_cols:
                    features = data[feature_cols].values
                    labels = data[label_col].tolist() if label_col else [None] * len(data)
                    return None, labels, features
                else:
                    # Use all numeric columns as features
                    numeric_cols = data.select_dtypes(include=[np.number]).columns
                    features = data[numeric_cols].values
                    return None, [None] * len(data), features
            
            else:
                # Unknown format - try to convert
                print(f"Unknown format: {type(data)}, attempting conversion...")
                try:
                    features = np.array(data)
                    if len(features.shape) > 1:
                        return None, [None] * features.shape[0], features
                    else:
                        return None, features.tolist(), None
                except:
                    raise ValueError(f"Cannot handle pickle format: {type(data)}")
                    
        except Exception as e:
            print(f"Error loading pickle file: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _load_data(self):
        """Load lyrics dataset from CSV files."""
        data_files = []
        labels = []
        
        if os.path.isdir(self.data_path):
            for root, dirs, files in os.walk(self.data_path):
                for file in files:
                    if file.endswith('.csv'):
                        file_path = os.path.join(root, file)
                        try:
                            df = pd.read_csv(file_path)
                            text_col = None
                            lang_col = None
                            
                            for col in df.columns:
                                col_lower = col.lower()
                                if 'lyric' in col_lower or 'text' in col_lower or 'song' in col_lower:
                                    text_col = col
                                if 'language' in col_lower or 'lang' in col_lower:
                                    lang_col = col
                            
                            if text_col:
                                data_files.extend(df[text_col].astype(str).tolist())
                                if lang_col:
                                    labels.extend(df[lang_col].tolist())
                                else:
                                    labels.extend([None] * len(df))
                        except Exception as e:
                            print(f"Error loading {file_path}: {e}")
        elif os.path.isfile(self.data_path) and self.data_path.endswith('.csv'):
            # Single CSV file
            try:
                df = pd.read_csv(self.data_path)
                text_col = None
                lang_col = None
                
                for col in df.columns:
                    col_lower = col.lower()
                    if 'lyric' in col_lower or 'text' in col_lower or 'song' in col_lower:
                        text_col = col
                    if 'language' in col_lower or 'lang' in col_lower:
                        lang_col = col
                
                if text_col:
                    data_files = df[text_col].astype(str).tolist()
                    labels = df[lang_col].tolist() if lang_col else [None] * len(df)
            except Exception as e:
                print(f"Error loading CSV file: {e}")
        
        return data_files, labels
    
    def _extract_features(self):
        """Extract TF-IDF features from lyrics."""
        if self.data is None or len(self.data) == 0:
            raise ValueError("No data loaded. Check dataset path.")
        
        print(f"Extracting TF-IDF features from {len(self.data)} text samples...")
        
        # Convert all to strings and filter empty
        text_data = []
        valid_indices = []
        for i, item in enumerate(self.data):
            if item is not None:
                text = str(item).strip()
                if len(text) > 0:
                    text_data.append(text)
                    valid_indices.append(i)
        
        if len(text_data) == 0:
            raise ValueError("No valid text data found after filtering.")
        
        if len(text_data) < len(self.data):
            print(f"Filtered {len(self.data) - len(text_data)} empty/invalid samples")
            # Update labels to match valid indices
            if self.labels and len(self.labels) == len(self.data):
                self.labels = [self.labels[i] for i in valid_indices]
            self.data = text_data
        
        print(f"Using {len(text_data)} valid text samples for TF-IDF extraction...")
        
        vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            lowercase=True
        )
        
        try:
            features = vectorizer.fit_transform(text_data).toarray()
            self.vectorizer = vectorizer
            
            print(f"Extracted {features.shape[1]} TF-IDF features from {features.shape[0]} samples")
            return features
        except Exception as e:
            print(f"Error in TF-IDF extraction: {e}")
            raise ValueError(f"Failed to extract TF-IDF features: {e}")
    
    def __len__(self):
        if self.features is not None:
            return len(self.features)
        elif self.data is not None:
            return len(self.data)
        else:
            return 0
    
    def __getitem__(self, idx):
        if self.features is not None:
            feature = torch.FloatTensor(self.features[idx])
            label = self.labels[idx] if idx < len(self.labels) and self.labels[idx] is not None else -1
            return feature, label
        else:
            raise ValueError("No features available. Check dataset loading.")


def get_dataloader(data_path, batch_size=32, shuffle=True, max_features=5000, is_pickle=False, max_samples=1000):
    
    dataset = LyricsDataset(data_path, max_features=max_features, is_pickle=is_pickle, max_samples=max_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return dataloader, dataset


# ============================================================================
# VAE MODELS (Same as before)
# ============================================================================
class VAE(nn.Module):
    """Variational Autoencoder for learning latent representations."""
    
    def __init__(self, input_dim, latent_dim=32, hidden_dims=[512, 256]):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


class BetaVAE(VAE):
    """Beta-VAE for disentangled representations."""
    
    def __init__(self, input_dim, latent_dim=32, hidden_dims=[512, 256], beta=4.0):
        super(BetaVAE, self).__init__(input_dim, latent_dim, hidden_dims)
        self.beta = beta


class ConditionalVAE(nn.Module):
    """Conditional VAE with label conditioning."""
    
    def __init__(self, input_dim, latent_dim=32, condition_dim=5, hidden_dims=[512, 256]):
        super(ConditionalVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        
        # Encoder (takes input + condition)
        encoder_layers = []
        prev_dim = input_dim + condition_dim
        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        
        # Decoder (takes latent + condition)
        decoder_layers = []
        prev_dim = latent_dim + condition_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x, c):
        x_c = torch.cat([x, c], dim=1)
        h = self.encoder(x_c)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, c):
        z_c = torch.cat([z, c], dim=1)
        return self.decoder(z_c)
    
    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, c)
        return recon, mu, logvar


class Autoencoder(nn.Module):
    """Non-variational Autoencoder baseline."""
    
    def __init__(self, input_dim, latent_dim=32, hidden_dims=[512, 256]):
        super(Autoencoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z, None


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================
def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """VAE loss function."""
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (BCE + beta * KLD) / x.size(0), BCE / x.size(0), beta * KLD / x.size(0)


def train_vae_model(model, dataloader, epochs=50, lr=1e-3, device='cpu', beta=1.0):
    """Train VAE model."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    history = {'loss': [], 'recon_loss': [], 'kl_loss': []}
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_recon = 0
        total_kl = 0
        
        for batch_idx, (data, _) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
            data = data.to(device)
            optimizer.zero_grad()
            
            recon_batch, mu, logvar = model(data)
            loss, recon_loss, kl_loss = vae_loss(recon_batch, data, mu, logvar, beta)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
        
        avg_loss = total_loss / len(dataloader)
        avg_recon = total_recon / len(dataloader)
        avg_kl = total_kl / len(dataloader)
        
        history['loss'].append(avg_loss)
        history['recon_loss'].append(avg_recon)
        history['kl_loss'].append(avg_kl)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Recon: {avg_recon:.4f}, KL: {avg_kl:.4f}")
    
    return history


def train_cvae_model(model, dataloader, epochs=50, lr=1e-3, device='cpu'):
    """Train Conditional VAE model."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    history = {'loss': [], 'recon_loss': [], 'kl_loss': []}
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_recon = 0
        total_kl = 0
        
        for batch_idx, (data, labels) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
            data = data.to(device)
            
            # Create one-hot condition vectors
            if labels[0] == -1:
                # No labels available, use dummy condition
                condition = torch.zeros(len(data), model.condition_dim).to(device)
            else:
                condition = F.one_hot(torch.clamp(labels.long(), 0, model.condition_dim-1), 
                                     model.condition_dim).float().to(device)
            
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data, condition)
            loss, recon_loss, kl_loss = vae_loss(recon_batch, data, mu, logvar)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
        
        avg_loss = total_loss / len(dataloader)
        avg_recon = total_recon / len(dataloader)
        avg_kl = total_kl / len(dataloader)
        
        history['loss'].append(avg_loss)
        history['recon_loss'].append(avg_recon)
        history['kl_loss'].append(avg_kl)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Recon: {avg_recon:.4f}, KL: {avg_kl:.4f}")
    
    return history


def train_autoencoder(model, dataloader, epochs=50, lr=1e-3, device='cpu'):
    """Train Autoencoder model."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    history = {'loss': []}
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (data, _) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
            data = data.to(device)
            optimizer.zero_grad()
            
            recon_batch, _, _ = model(data)
            loss = criterion(recon_batch, data)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        history['loss'].append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
    
    return history


# ============================================================================
# CLUSTERING FUNCTIONS
# ============================================================================
def extract_latent_representations(model, dataloader, device='cpu', is_cvae=False):
    """Extract latent representations from trained model."""
    model.eval()
    latent_reprs = []
    labels = []
    
    with torch.no_grad():
        for data, label in dataloader:
            data = data.to(device)
            if is_cvae:
                # For CVAE, use dummy condition
                condition = torch.zeros(len(data), model.condition_dim).to(device)
                mu, _ = model.encode(data, condition)
                z = mu
            else:
                if isinstance(model, Autoencoder):
                    z = model.encode(data)
                else:
                    mu, _ = model.encode(data)
                    z = mu
            
            latent_reprs.append(z.cpu().numpy())
            labels.extend(label.tolist())
    
    return np.vstack(latent_reprs), np.array(labels)


def kmeans_clustering(latent_reprs, n_clusters=5, random_state=42):
    """K-Means clustering."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(latent_reprs)
    return labels, kmeans


def agglomerative_clustering(latent_reprs, n_clusters=5):
    """Agglomerative clustering."""
    agg = AgglomerativeClustering(n_clusters=n_clusters)
    labels = agg.fit_predict(latent_reprs)
    return labels, agg


def dbscan_clustering(latent_reprs, eps=0.5, min_samples=5):
    """DBSCAN clustering."""
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(latent_reprs)
    return labels, dbscan


def pca_baseline(features, n_components=32):
    """PCA dimensionality reduction."""
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(features)
    return reduced, pca


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================
def cluster_purity(y_true, y_pred):
    """Calculate cluster purity."""
    n_clusters = len(np.unique(y_pred))
    n_samples = len(y_true)
    
    purity = 0
    for k in range(n_clusters):
        cluster_indices = np.where(y_pred == k)[0]
        if len(cluster_indices) == 0:
            continue
        
        cluster_labels = y_true[cluster_indices]
        if len(cluster_labels) > 0:
            most_common = np.bincount(cluster_labels).argmax()
            purity += np.sum(cluster_labels == most_common)
    
    return purity / n_samples


def evaluate_clustering(latent_reprs, cluster_labels, true_labels=None):
    """Evaluate clustering quality."""
    metrics = {}
    
    # Unsupervised metrics
    if len(np.unique(cluster_labels)) > 1:
        metrics['silhouette'] = silhouette_score(latent_reprs, cluster_labels)
        metrics['calinski_harabasz'] = calinski_harabasz_score(latent_reprs, cluster_labels)
        metrics['davies_bouldin'] = davies_bouldin_score(latent_reprs, cluster_labels)
    
    # Supervised metrics (if labels available)
    if true_labels is not None and len(np.unique(true_labels)) > 1:
        valid_mask = true_labels != -1
        if np.sum(valid_mask) > 0:
            metrics['ari'] = adjusted_rand_score(true_labels[valid_mask], cluster_labels[valid_mask])
            metrics['nmi'] = normalized_mutual_info_score(true_labels[valid_mask], cluster_labels[valid_mask])
            metrics['purity'] = cluster_purity(true_labels[valid_mask], cluster_labels[valid_mask])
    
    return metrics


def print_metrics(metrics, title="Clustering Metrics"):
    """Print metrics in formatted way."""
    print(f"\n{title}:")
    print("=" * 50)
    for key, value in metrics.items():
        print(f"{key:20s}: {value:.4f}")


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================
def plot_tsne(latent_reprs, labels, title="t-SNE Visualization"):
    """Plot t-SNE visualization."""
    tsne = TSNE(n_components=2, random_state=42)
    reduced = tsne.fit_transform(latent_reprs)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.tight_layout()
    plt.show()


def plot_umap(latent_reprs, labels, title="UMAP Visualization"):
    """Plot UMAP visualization."""
    if not UMAP_AVAILABLE:
        print("UMAP not available. Skipping UMAP visualization.")
        return
    
    reducer = umap.UMAP(n_components=2, random_state=42)
    reduced = reducer.fit_transform(latent_reprs)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel("UMAP Component 1")
    plt.ylabel("UMAP Component 2")
    plt.tight_layout()
    plt.show()


def plot_training_history(history, title="Training History"):
    """Plot training history."""
    fig, axes = plt.subplots(1, len(history), figsize=(5*len(history), 4))
    if len(history) == 1:
        axes = [axes]
    
    for idx, (key, values) in enumerate(history.items()):
        axes[idx].plot(values)
        axes[idx].set_title(f"{key.replace('_', ' ').title()}")
        axes[idx].set_xlabel("Epoch")
        axes[idx].set_ylabel("Loss")
        axes[idx].grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


# ============================================================================
# TASK FUNCTIONS
# ============================================================================
def run_easy_task(data_path, epochs=50, batch_size=32, max_features=5000, 
                  latent_dim=32, n_clusters=5, is_pickle=False, max_samples=1000):
    """Run Easy Task: Basic VAE + K-Means."""
    print("=" * 60)
    print("EASY TASK: Basic VAE + K-Means")
    print("=" * 60)
    print(f"Using up to {max_samples} samples")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load dataset
    print("\n[1/6] Loading dataset...")
    dataloader, dataset = get_dataloader(data_path, batch_size=batch_size, 
                                        max_features=max_features, is_pickle=is_pickle,
                                        max_samples=max_samples)
    input_dim = dataset.features.shape[1]
    print(f"Dataset loaded: {len(dataset)} samples, {input_dim} features")
    
    # Train VAE
    print("\n[2/6] Training VAE...")
    vae = VAE(input_dim, latent_dim=latent_dim)
    history = train_vae_model(vae, dataloader, epochs=epochs, device=device)
    
    # Extract latent representations
    print("\n[3/6] Extracting latent representations...")
    latent_reprs, true_labels = extract_latent_representations(vae, dataloader, device=device)
    print(f"Latent representations shape: {latent_reprs.shape}")
    
    # K-Means clustering
    print("\n[4/6] Performing K-Means clustering...")
    cluster_labels, kmeans = kmeans_clustering(latent_reprs, n_clusters=n_clusters)
    print(f"Found {len(np.unique(cluster_labels))} clusters")
    
    # PCA baseline
    print("\n[5/6] Computing PCA baseline...")
    pca_reprs, pca_model = pca_baseline(dataset.features, n_components=latent_dim)
    pca_cluster_labels, _ = kmeans_clustering(pca_reprs, n_clusters=n_clusters)
    
    # Evaluation
    print("\n[6/6] Evaluating results...")
    vae_metrics = evaluate_clustering(latent_reprs, cluster_labels, true_labels)
    pca_metrics = evaluate_clustering(pca_reprs, pca_cluster_labels, true_labels)
    
    print_metrics(vae_metrics, "VAE + K-Means Metrics")
    print_metrics(pca_metrics, "PCA + K-Means Metrics")
    
    # Visualization
    print("\nGenerating visualizations...")
    plot_training_history(history)
    plot_tsne(latent_reprs, cluster_labels, "VAE Latent Space (t-SNE)")
    if UMAP_AVAILABLE:
        plot_umap(latent_reprs, cluster_labels, "VAE Latent Space (UMAP)")
    
    return vae, latent_reprs, cluster_labels, vae_metrics


def run_medium_task(data_path, epochs=50, batch_size=32, max_features=5000,
                   latent_dim=32, n_clusters=5, is_pickle=False, max_samples=1000):
    """Run Medium Task: Enhanced VAE + Multiple Clustering Algorithms."""
    print("=" * 60)
    print("MEDIUM TASK: Enhanced VAE + Multiple Clustering Algorithms")
    print("=" * 60)
    print(f"Using up to {max_samples} samples")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load dataset
    print("\n[1/7] Loading dataset...")
    dataloader, dataset = get_dataloader(data_path, batch_size=batch_size,
                                        max_features=max_features, is_pickle=is_pickle,
                                        max_samples=max_samples)
    input_dim = dataset.features.shape[1]
    print(f"Dataset loaded: {len(dataset)} samples, {input_dim} features")
    
    # Train enhanced VAE
    print("\n[2/7] Training enhanced VAE...")
    vae = VAE(input_dim, latent_dim=latent_dim, hidden_dims=[512, 256, 128])
    history = train_vae_model(vae, dataloader, epochs=epochs, device=device)
    
    # Extract latent representations
    print("\n[3/7] Extracting latent representations...")
    latent_reprs, true_labels = extract_latent_representations(vae, dataloader, device=device)
    
    # Multiple clustering algorithms
    print("\n[4/7] Performing clustering with multiple algorithms...")
    
    # K-Means
    kmeans_labels, _ = kmeans_clustering(latent_reprs, n_clusters=n_clusters)
    print(f"K-Means: {len(np.unique(kmeans_labels))} clusters")
    
    # Agglomerative
    agg_labels, _ = agglomerative_clustering(latent_reprs, n_clusters=n_clusters)
    print(f"Agglomerative: {len(np.unique(agg_labels))} clusters")
    
    # DBSCAN
    dbscan_labels, _ = dbscan_clustering(latent_reprs)
    n_dbscan_clusters = len(np.unique(dbscan_labels[dbscan_labels != -1]))
    print(f"DBSCAN: {n_dbscan_clusters} clusters (+ {np.sum(dbscan_labels == -1)} noise points)")
    
    # Evaluation
    print("\n[5/7] Evaluating clustering results...")
    kmeans_metrics = evaluate_clustering(latent_reprs, kmeans_labels, true_labels)
    agg_metrics = evaluate_clustering(latent_reprs, agg_labels, true_labels)
    dbscan_metrics = evaluate_clustering(latent_reprs, dbscan_labels, true_labels)
    
    print_metrics(kmeans_metrics, "K-Means Metrics")
    print_metrics(agg_metrics, "Agglomerative Metrics")
    print_metrics(dbscan_metrics, "DBSCAN Metrics")
    
    # Visualization
    print("\n[6/7] Generating visualizations...")
    plot_training_history(history)
    plot_tsne(latent_reprs, kmeans_labels, "K-Means Clustering (t-SNE)")
    plot_tsne(latent_reprs, agg_labels, "Agglomerative Clustering (t-SNE)")
    plot_tsne(latent_reprs, dbscan_labels, "DBSCAN Clustering (t-SNE)")
    
    # Comparison
    print("\n[7/7] Clustering comparison summary:")
    all_metrics = {
        'K-Means': kmeans_metrics,
        'Agglomerative': agg_metrics,
        'DBSCAN': dbscan_metrics
    }
    
    for name, metrics in all_metrics.items():
        print(f"\n{name}:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
    
    return vae, latent_reprs, all_metrics


def run_hard_task(data_path, epochs=50, batch_size=32, max_features=5000,
                 latent_dim=32, n_clusters=5, beta=4.0, is_pickle=False, max_samples=1000):
    """Run Hard Task: CVAE/Beta-VAE + Comprehensive Comparison."""
    print("=" * 60)
    print("HARD TASK: CVAE/Beta-VAE + Comprehensive Comparison")
    print("=" * 60)
    print(f"Using up to {max_samples} samples")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load dataset
    print("\n[1/10] Loading dataset...")
    dataloader, dataset = get_dataloader(data_path, batch_size=batch_size,
                                        max_features=max_features, is_pickle=is_pickle,
                                        max_samples=max_samples)
    input_dim = dataset.features.shape[1]
    true_labels_array = np.array([dataset.labels[i] if dataset.labels[i] is not None else -1 
                                  for i in range(len(dataset))])
    print(f"Dataset loaded: {len(dataset)} samples, {input_dim} features")
    
    # Determine condition dimension for CVAE
    unique_labels = np.unique(true_labels_array[true_labels_array != -1])
    condition_dim = max(len(unique_labels), 5) if len(unique_labels) > 0 else 5
    
    # Train multiple models
    models = {}
    histories = {}
    latent_reprs_dict = {}
    
    # Beta-VAE
    print("\n[2/10] Training Beta-VAE...")
    beta_vae = BetaVAE(input_dim, latent_dim=latent_dim, beta=beta)
    histories['Beta-VAE'] = train_vae_model(beta_vae, dataloader, epochs=epochs, 
                                           device=device, beta=beta)
    models['Beta-VAE'] = beta_vae
    latent_reprs_dict['Beta-VAE'], _ = extract_latent_representations(beta_vae, dataloader, device=device)
    
    # CVAE
    print("\n[3/10] Training Conditional VAE...")
    cvae = ConditionalVAE(input_dim, latent_dim=latent_dim, condition_dim=condition_dim)
    histories['CVAE'] = train_cvae_model(cvae, dataloader, epochs=epochs, device=device)
    models['CVAE'] = cvae
    latent_reprs_dict['CVAE'], _ = extract_latent_representations(cvae, dataloader, device=device, is_cvae=True)
    
    # Autoencoder
    print("\n[4/10] Training Autoencoder...")
    ae = Autoencoder(input_dim, latent_dim=latent_dim)
    histories['Autoencoder'] = train_autoencoder(ae, dataloader, epochs=epochs, device=device)
    models['Autoencoder'] = ae
    latent_reprs_dict['Autoencoder'], _ = extract_latent_representations(ae, dataloader, device=device)
    
    # PCA Baseline
    print("\n[5/10] Computing PCA baseline...")
    pca_reprs, _ = pca_baseline(dataset.features, n_components=latent_dim)
    latent_reprs_dict['PCA'] = pca_reprs
    
    # Clustering and evaluation
    print("\n[6/10] Performing clustering and evaluation...")
    all_results = {}
    
    for model_name, latent_reprs in latent_reprs_dict.items():
        print(f"\nEvaluating {model_name}...")
        cluster_labels, _ = kmeans_clustering(latent_reprs, n_clusters=n_clusters)
        metrics = evaluate_clustering(latent_reprs, cluster_labels, true_labels_array)
        all_results[model_name] = {
            'latent_reprs': latent_reprs,
            'cluster_labels': cluster_labels,
            'metrics': metrics
        }
        print_metrics(metrics, f"{model_name} Metrics")
    
    # Visualizations
    print("\n[7/10] Generating visualizations...")
    for name in histories.keys():
        plot_training_history(histories[name], f"{name} Training History")
    
    for model_name, results in all_results.items():
        plot_tsne(results['latent_reprs'], results['cluster_labels'], 
                 f"{model_name} Clustering (t-SNE)")
    
    # Comparison table
    print("\n[8/10] Comprehensive comparison:")
    print("=" * 80)
    print(f"{'Method':<20} {'Silhouette':<12} {'CH Index':<12} {'DB Index':<12} "
          f"{'ARI':<12} {'NMI':<12} {'Purity':<12}")
    print("=" * 80)
    
    for model_name, results in all_results.items():
        m = results['metrics']
        print(f"{model_name:<20} "
              f"{m.get('silhouette', 0):>10.4f}  "
              f"{m.get('calinski_harabasz', 0):>10.4f}  "
              f"{m.get('davies_bouldin', 0):>10.4f}  "
              f"{m.get('ari', 0):>10.4f}  "
              f"{m.get('nmi', 0):>10.4f}  "
              f"{m.get('purity', 0):>10.4f}")
    
    # Best method identification
    print("\n[9/10] Identifying best method...")
    best_method = None
    best_score = -1
    
    for model_name, results in all_results.items():
        m = results['metrics']
        # Use silhouette score as primary metric
        score = m.get('silhouette', -1)
        if score > best_score:
            best_score = score
            best_method = model_name
    
    print(f"Best method (by Silhouette Score): {best_method} ({best_score:.4f})")
    
    # Summary
    print("\n[10/10] Task complete!")
    print("=" * 60)
    
    return models, all_results, best_method


# ============================================================================
# MAIN EXECUTION (for Google Colab)
# ============================================================================
# ============================================================================
# DEFAULT DATASET PATH (Google Colab)
# ============================================================================
# Try extracted dataset first (if extraction was run), otherwise use original
DEFAULT_PKL_PATH = "/content/dataset_extracted.pkl"  # Use extracted version if available
# If extracted doesn't exist, fallback to original
import os
if not os.path.exists(DEFAULT_PKL_PATH):
    DEFAULT_PKL_PATH = "/content/dataset_positive_256_clean.pkl"

# ============================================================================
# AUTO-RUN ON IMPORT (For Google Colab)
# ============================================================================
# Set AUTO_RUN = True to automatically run ALL tasks when code is loaded
AUTO_RUN = True  # Change to False if you want to run tasks manually

if AUTO_RUN:
    print("=" * 60)
    print("VAE Music Clustering Project - All in One with 1000 Sample Limit")
    print("=" * 60)
    print(f"Dataset path: {DEFAULT_PKL_PATH}")
    print("Using 1000 samples maximum")
    print("Will run ALL tasks automatically: Easy → Medium → Hard")
    print("=" * 60)
    
    try:
        # ====================================================================
        # TASK 1: EASY TASK
        # ====================================================================
        print("\n" + "=" * 60)
        print("STARTING TASK 1/3: EASY TASK")
        print("=" * 60)
        print("Basic VAE + K-Means Clustering")
        print("=" * 60)
        
        run_easy_task(
            data_path=DEFAULT_PKL_PATH,
            epochs=50,
            batch_size=32,
            max_features=5000,
            latent_dim=32,
            n_clusters=5,
            is_pickle=True,
            max_samples=1000
        )
        
        print("\n" + "=" * 60)
        print("TASK 1 COMPLETED: Easy Task")
        print("=" * 60)
        
        # ====================================================================
        # TASK 2: MEDIUM TASK
        # ====================================================================
        print("\n" + "=" * 60)
        print("STARTING TASK 2/3: MEDIUM TASK")
        print("=" * 60)
        print("Enhanced VAE + Multiple Clustering Algorithms")
        print("=" * 60)
        
        run_medium_task(
            data_path=DEFAULT_PKL_PATH,
            epochs=50,
            batch_size=32,
            max_features=5000,
            latent_dim=32,
            n_clusters=5,
            is_pickle=True,
            max_samples=1000
        )
        
        print("\n" + "=" * 60)
        print("TASK 2 COMPLETED: Medium Task")
        print("=" * 60)
        
        # ====================================================================
        # TASK 3: HARD TASK
        # ====================================================================
        print("\n" + "=" * 60)
        print("STARTING TASK 3/3: HARD TASK")
        print("=" * 60)
        print("CVAE/Beta-VAE + Comprehensive Comparison")
        print("=" * 60)
        
        run_hard_task(
            data_path=DEFAULT_PKL_PATH,
            epochs=50,
            batch_size=32,
            max_features=5000,
            latent_dim=32,
            n_clusters=5,
            beta=4.0,
            is_pickle=True,
            max_samples=1000
        )
        
        print("\n" + "=" * 60)
        print("TASK 3 COMPLETED: Hard Task")
        print("=" * 60)
        
        # ====================================================================
        # FINAL SUMMARY
        # ====================================================================
        print("\n" + "=" * 60)
        print("ALL TASKS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("Completed:")
        print("  ✓ Easy Task: Basic VAE + K-Means")
        print("  ✓ Medium Task: Enhanced VAE + Multiple Clustering")
        print("  ✓ Hard Task: CVAE/Beta-VAE + Comprehensive Comparison")
        print("=" * 60)
        print("All results and visualizations have been generated.")
        print("=" * 60)
        
    except FileNotFoundError:
        print(f"\nERROR: Dataset not found at {DEFAULT_PKL_PATH}")
        print("Please upload your dataset to /content/ directory")
        print("\nManual usage:")
        print("  run_easy_task('/content/dataset_positive_256_clean.pkl', is_pickle=True)")
        print("  run_medium_task('/content/dataset_positive_256_clean.pkl', is_pickle=True)")
        print("  run_hard_task('/content/dataset_positive_256_clean.pkl', is_pickle=True)")
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nPlease check your dataset path and try again.")
        print("Or set AUTO_RUN = False and run tasks manually:")

elif __name__ == "__main__":
    print("VAE Music Clustering Project - All in One with 1000 Sample Limit")
    print("=" * 60)
    print(f"\nDefault dataset path: {DEFAULT_PKL_PATH}")
    print("Default: Uses 1000 samples maximum")
    print("\nQuick start with default path:")
    print("  run_easy_task(DEFAULT_PKL_PATH, is_pickle=True)")
    print("  run_medium_task(DEFAULT_PKL_PATH, is_pickle=True)")
    print("  run_hard_task(DEFAULT_PKL_PATH, is_pickle=True)")
    print("\nOr use the path directly:")
    print("  run_easy_task('/content/dataset_positive_256_clean.pkl', is_pickle=True)")
    print("\n" + "=" * 60)

