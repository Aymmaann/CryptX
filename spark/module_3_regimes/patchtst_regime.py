from pyspark.sql import DataFrame
from pyspark.sql import functions as F
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class PatchTSTRegimeDetector:
    """
    PatchTST-based regime detector with UNSUPERVISED deep clustering.
    
    Instead of training on pseudo-labels (which just mimics threshold),
    we use a two-step approach:
    
    1. Train a PatchTST autoencoder to learn temporal representations
    2. Apply K-Means clustering on the learned embeddings
    
    This lets the model discover regime patterns beyond simple volatility quantiles.
    """

    def __init__(self, n_regimes=3, seq_len=240, patch_len=16,
                 d_model=64, n_heads=4, n_layers=2,
                 epochs=15, batch_size=256, lr=1e-3, random_state=42):
        """
        Args:
            seq_len: Increased to 240 (4 hours) for better context
            Other params same as before
        """
        self.n_regimes    = n_regimes
        self.seq_len      = seq_len
        self.patch_len    = patch_len
        self.d_model      = d_model
        self.n_heads      = n_heads
        self.n_layers     = n_layers
        self.epochs       = epochs
        self.batch_size   = batch_size
        self.lr           = lr
        self.random_state = random_state
        self.scaler       = StandardScaler()
        self.model        = None
        self.kmeans       = None
        self.device       = torch.device('cpu')

        torch.manual_seed(random_state)
        np.random.seed(random_state)

        print(f"[INFO] PatchTST (Unsupervised) initialized | seq_len={seq_len} "
              f"patch_len={patch_len} d_model={d_model}")

    def detect(self, df: DataFrame) -> DataFrame:
        print("[PATCHTST] Starting UNSUPERVISED regime detection...")

        feature_cols = [
            'log_return', 'vol_60m', 'vol_240m', 'vol_ratio_60_240',
            'cum_return_60m', 'return_skew_60m',
            'volume_spike', 'rsi', 'bb_width', 'atr_pct'
        ]
        available = [c for c in feature_cols if c in df.columns]
        print(f"[PATCHTST] Using {len(available)} features")

        print("[PATCHTST] Converting to Pandas...")
        df_pd = (df.select(['timestamp'] + available)
                   .toPandas()
                   .sort_values('timestamp')
                   .reset_index(drop=True))

        X_raw = df_pd[available].values
        X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)

        # Scale
        print("[PATCHTST] Scaling features...")
        X_scaled = self.scaler.fit_transform(X_raw)

        # Build sequences
        print("[PATCHTST] Building sequences...")
        X_seq, idx_seq = self._build_sequences(X_scaled)
        print(f"[PATCHTST] {X_seq.shape[0]:,} sequences")

        # Train autoencoder (unsupervised)
        n_patches = self.seq_len // self.patch_len
        self.model = _PatchTSTAutoencoder(
            n_features = len(available),
            patch_len  = self.patch_len,
            n_patches  = n_patches,
            d_model    = self.d_model,
            n_heads    = self.n_heads,
            n_layers   = self.n_layers
        ).to(self.device)

        print(f"[PATCHTST] Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        self._train_autoencoder(X_seq)

        # Extract embeddings
        print("[PATCHTST] Extracting embeddings...")
        embeddings = self._extract_embeddings(X_seq)

        # Cluster embeddings
        print(f"[PATCHTST] Clustering embeddings into {self.n_regimes} regimes...")
        self.kmeans = KMeans(n_clusters=self.n_regimes, 
                            random_state=self.random_state, n_init=20)
        cluster_labels = self.kmeans.fit_predict(embeddings)

        # Map back to timesteps
        regimes = np.full(len(df_pd), -1, dtype=int)
        for label, idx in zip(cluster_labels, idx_seq):
            regimes[idx] = label

        first_valid = regimes[regimes >= 0][0]
        regimes[regimes == -1] = first_valid

        # Map by volatility
        vol_idx  = available.index('vol_60m') if 'vol_60m' in available else 0
        vol_vals = X_raw[:, vol_idx]
        regimes = self._map_regimes_by_volatility(regimes, vol_vals)

        df_pd['regime']      = regimes
        df_pd['regime_name'] = df_pd['regime'].map(
            {0: 'low_vol', 1: 'medium_vol', 2: 'high_vol'}
        )

        print("[PATCHTST] Converting back to Spark...")
        spark     = df.sparkSession
        df_result = spark.createDataFrame(
            df_pd[['timestamp', 'regime', 'regime_name']]
        )
        df_result = df.join(df_result, on='timestamp', how='inner')
        df_result = df_result.cache()
        df_result.count()

        print("[PATCHTST] Detection complete")
        return df_result

    def _build_sequences(self, X):
        """Build sequences without labels."""
        seqs, indices = [], []
        for i in range(self.seq_len, len(X)):
            seqs.append(X[i - self.seq_len: i])
            indices.append(i)
        return np.array(seqs, dtype=np.float32), indices

    def _train_autoencoder(self, X_seq):
        """Train autoencoder to learn representations."""
        # Subsample
        n_train = min(50000, len(X_seq))
        indices = np.random.choice(len(X_seq), n_train, replace=False)
        X_train = X_seq[indices]
        
        print(f"[PATCHTST] Training autoencoder on {n_train:,} sequences...")
        
        loader    = DataLoader(_AutoencoderDataset(X_train),
                               batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        self.model.train()
        best_loss = float('inf')
        patience = 0

        for epoch in range(1, self.epochs + 1):
            total_loss = 0.0
            for xb in loader:
                optimizer.zero_grad()
                recon = self.model(xb)
                loss = nn.MSELoss()(recon, xb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item() * len(xb)

            avg_loss = total_loss / len(X_train)
            
            if epoch % 5 == 0 or epoch == 1:
                print(f"[PATCHTST] Epoch {epoch:3d}/{self.epochs} | loss={avg_loss:.6f}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience = 0
            else:
                patience += 1
            
            if patience >= 3 and epoch >= 8:
                print(f"[PATCHTST] Early stopping at epoch {epoch}")
                break

        print("[PATCHTST] Autoencoder training complete")

    def _extract_embeddings(self, X_seq):
        """Extract embeddings from trained encoder."""
        self.model.eval()
        loader = DataLoader(_AutoencoderDataset(X_seq),
                           batch_size=self.batch_size * 2, shuffle=False)
        embeddings = []
        
        with torch.no_grad():
            for xb in loader:
                emb = self.model.encode(xb)
                embeddings.append(emb.numpy())
        
        return np.vstack(embeddings)

    def _map_regimes_by_volatility(self, regimes, volatility):
        regime_vols = {
            r: volatility[regimes == r].mean()
            for r in range(self.n_regimes)
            if (regimes == r).sum() > 0
        }
        regime_map = {old: new for new, old
                      in enumerate(sorted(regime_vols, key=regime_vols.get))}
        return np.array([regime_map.get(r, r) for r in regimes])


# ======================================================================
# Autoencoder Architecture
# ======================================================================

class _PatchTSTAutoencoder(nn.Module):
    """PatchTST as an autoencoder for unsupervised learning."""
    
    def __init__(self, n_features, patch_len, n_patches,
                 d_model, n_heads, n_layers):
        super().__init__()
        self.patch_len = patch_len
        self.n_patches = n_patches
        self.n_features = n_features

        # Encoder
        self.patch_embed = nn.Linear(patch_len * n_features, d_model)
        self.pos_embed   = nn.Parameter(torch.randn(1, n_patches, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Decoder
        self.decoder = nn.Linear(d_model, patch_len * n_features)

    def encode(self, x):
        """Encode to embeddings."""
        B, T, C = x.shape
        x = x.reshape(B, self.n_patches, self.patch_len * C)
        x = self.patch_embed(x) + self.pos_embed
        x = self.transformer(x)
        return x.mean(dim=1)  # Global average pooling

    def forward(self, x):
        """Full forward pass (encode + decode)."""
        B, T, C = x.shape
        # Encode
        patches = x.reshape(B, self.n_patches, self.patch_len * C)
        emb = self.patch_embed(patches) + self.pos_embed
        encoded = self.transformer(emb)
        # Decode
        decoded = self.decoder(encoded)
        # Reshape back
        return decoded.reshape(B, T, C)


class _AutoencoderDataset(Dataset):
    def __init__(self, X):
        self.X = torch.tensor(X, dtype=torch.float32)
    def __len__(self):  return len(self.X)
    def __getitem__(self, i): return self.X[i]