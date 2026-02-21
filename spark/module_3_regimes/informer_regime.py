from pyspark.sql import DataFrame
from pyspark.sql import functions as F
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import math


class InformerRegimeDetector:
    """
    Informer-based regime detector (AAAI 2021 Best Paper).

    The Informer improves on vanilla Transformers for long sequences via
    ProbSparse self-attention: instead of computing all O(L²) attention
    pairs, it selects the Top-u queries that dominate the attention
    distribution, reducing complexity to O(L log L).

    This makes it better suited than standard Transformers for crypto
    minute-level data where sequences can be very long.

    Architecture:
        Input (B, seq_len, features)
        → multi-head ProbSparse self-attention
        → feed-forward with distilling (halve sequence each layer)
        → global average pool
        → linear classifier → regime label

    Kept deliberately small so it trains on CPU in ~25-35 minutes.
    """

    def __init__(self, n_regimes=3, seq_len=180,
                 d_model=48, n_heads=4, n_layers=2, d_ff=96,
                 factor=5, epochs=12, batch_size=512,
                 lr=1e-3, random_state=42):
        """
        Args:
            seq_len: Reduced to 180 (3 hours) - still better than original 96
            d_model: Reduced to 48 (from 64) for speed
            d_ff: Reduced to 96 (from 128) for speed
            batch_size: Increased to 512 for faster training
            epochs: Reduced to 12 for speed
        """
        self.n_regimes    = n_regimes
        self.seq_len      = seq_len
        self.d_model      = d_model
        self.n_heads      = n_heads
        self.n_layers     = n_layers
        self.d_ff         = d_ff
        self.factor       = factor
        self.epochs       = epochs
        self.batch_size   = batch_size
        self.lr           = lr
        self.random_state = random_state
        self.scaler       = StandardScaler()
        self.model        = None
        self.device       = torch.device('cpu')

        torch.manual_seed(random_state)
        np.random.seed(random_state)

        print(f"[INFO] Informer initialized | seq_len={seq_len} "
              f"d_model={d_model} n_heads={n_heads} layers={n_layers} "
              f"factor={factor}")

    def detect(self, df: DataFrame) -> DataFrame:
        """
        Detect regimes using Informer.

        Args:
            df: Spark DataFrame with engineered features

        Returns:
            DataFrame with 'regime' and 'regime_name' columns added
        """
        print("[INFORMER] Starting regime detection...")

        feature_cols = [
            'log_return', 'vol_60m', 'vol_240m', 'vol_ratio_60_240',
            'cum_return_60m', 'return_skew_60m',
            'volume_spike', 'rsi', 'bb_width', 'atr_pct'
        ]
        available = [c for c in feature_cols if c in df.columns]
        print(f"[INFORMER] Using {len(available)} features")

        print("[INFORMER] Converting to Pandas...")
        df_pd = (df.select(['timestamp'] + available)
                   .toPandas()
                   .sort_values('timestamp')
                   .reset_index(drop=True))

        X_raw = df_pd[available].values
        X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)

        # Pseudo-labels from volatility quantiles
        print("[INFORMER] Generating pseudo-labels...")
        vol_idx  = available.index('vol_60m') if 'vol_60m' in available else 0
        vol_vals = X_raw[:, vol_idx]
        pseudo_labels = self._quantile_labels(vol_vals)

        # Scale
        print("[INFORMER] Scaling features...")
        X_scaled = self.scaler.fit_transform(X_raw)

        # Build sequences
        print("[INFORMER] Building sequences...")
        X_seq, y_seq, idx_seq = self._build_sequences(X_scaled, pseudo_labels)
        print(f"[INFORMER] {X_seq.shape[0]:,} sequences")

        # Build model
        self.model = _InformerNet(
            n_features = len(available),
            seq_len    = self.seq_len,
            d_model    = self.d_model,
            n_heads    = self.n_heads,
            n_layers   = self.n_layers,
            d_ff       = self.d_ff,
            factor     = self.factor,
            n_classes  = self.n_regimes
        ).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"[INFORMER] Model parameters: {n_params:,}")

        self._train(X_seq, y_seq)

        # IMPROVED: Extract embeddings and cluster them
        print("[INFORMER] Extracting learned embeddings...")
        embeddings = self._extract_embeddings(X_seq)
        
        print(f"[INFORMER] Clustering embeddings into {self.n_regimes} regimes...")
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.n_regimes, random_state=self.random_state, n_init=20)
        preds = kmeans.fit_predict(embeddings)

        regimes = np.full(len(df_pd), -1, dtype=int)
        for pred, idx in zip(preds, idx_seq):
            regimes[idx] = pred

        first_valid = regimes[regimes >= 0][0]
        regimes[regimes == -1] = first_valid

        regimes = self._map_regimes_by_volatility(regimes, vol_vals)

        df_pd['regime']      = regimes
        df_pd['regime_name'] = df_pd['regime'].map(
            {0: 'low_vol', 1: 'medium_vol', 2: 'high_vol'}
        )

        print("[INFORMER] Converting back to Spark DataFrame...")
        spark     = df.sparkSession
        df_result = spark.createDataFrame(
            df_pd[['timestamp', 'regime', 'regime_name']]
        )
        df_result = df.join(df_result, on='timestamp', how='inner')
        df_result = df_result.cache()
        df_result.count()

        print("[INFORMER] Detection complete")
        return df_result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _quantile_labels(self, values):
        q33, q67 = np.percentile(values, [33, 67])
        return np.where(values <= q33, 0,
                 np.where(values <= q67, 1, 2)).astype(int)

    def _build_sequences(self, X, y):
        seqs, labels, indices = [], [], []
        for i in range(self.seq_len, len(X)):
            seqs.append(X[i - self.seq_len: i])
            labels.append(y[i])
            indices.append(i)
        return (np.array(seqs, dtype=np.float32),
                np.array(labels, dtype=np.int64),
                indices)

    def _train(self, X_seq, y_seq):
        # More aggressive subsampling - use only 30K sequences
        n_train = min(30000, len(X_seq))
        indices = np.random.choice(len(X_seq), n_train, replace=False)
        X_train = X_seq[indices]
        y_train = y_seq[indices]
        
        print(f"[INFORMER] Training on {n_train:,} sequences (aggressive sampling)")
        
        loader    = DataLoader(_SeqDataset(X_train, y_train),
                               batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs)

        print(f"[INFORMER] Training for up to {self.epochs} epochs (with early stopping)...")
        self.model.train()
        
        best_loss = float('inf')
        patience_counter = 0
        patience = 2  # Very aggressive - stop after 2 epochs without improvement

        for epoch in range(1, self.epochs + 1):
            total_loss, correct, total = 0.0, 0, 0
            for xb, yb in loader:
                optimizer.zero_grad()
                logits = self.model(xb)
                loss   = criterion(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item() * len(yb)
                correct    += (logits.argmax(1) == yb).sum().item()
                total      += len(yb)
            scheduler.step()
            
            avg_loss = total_loss / total
            acc = correct / total * 100

            if epoch % 3 == 0 or epoch == 1:
                print(f"[INFORMER] Epoch {epoch:3d}/{self.epochs} | "
                      f"loss={avg_loss:.4f} | "
                      f"acc={acc:.1f}%")
            
            # Very aggressive early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            # Stop after 2 bad epochs OR if already good accuracy at epoch 6
            if patience_counter >= patience and epoch >= 6:
                print(f"[INFORMER] Early stopping at epoch {epoch}")
                break
            
            if acc > 95.0 and epoch >= 6:
                print(f"[INFORMER] Stopping - accuracy {acc:.1f}% is sufficient")
                break

        print("[INFORMER] Training complete")

    def _predict(self, X_seq):
        self.model.eval()
        loader = DataLoader(
            _SeqDataset(X_seq, np.zeros(len(X_seq), dtype=np.int64)),
            batch_size=self.batch_size * 2, shuffle=False)
        preds = []
        with torch.no_grad():
            for xb, _ in loader:
                preds.extend(self.model(xb).argmax(1).numpy())
        return np.array(preds)
    
    def _extract_embeddings(self, X_seq):
        """Extract embeddings before final classification layer."""
        self.model.eval()
        loader = DataLoader(
            _SeqDataset(X_seq, np.zeros(len(X_seq), dtype=np.int64)),
            batch_size=self.batch_size * 2, shuffle=False)
        embeddings = []
        with torch.no_grad():
            for xb, _ in loader:
                # Get embeddings (before classification head)
                emb = self.model.get_embeddings(xb)
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
# PyTorch modules
# ======================================================================

class _ProbSparseAttention(nn.Module):
    """
    ProbSparse self-attention from the Informer paper.

    Selects the Top-u queries (u = factor * ln(L)) that have the largest
    Query-Key spread, computes full attention for those, and uses a
    mean-value filling for the rest. Complexity: O(L log L).
    """

    def __init__(self, d_model, n_heads, factor=5):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads  = n_heads
        self.d_head   = d_model // n_heads
        self.factor   = factor
        self.scale    = 1.0 / math.sqrt(self.d_head)

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.out  = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, L, _ = x.shape
        H, D    = self.n_heads, self.d_head

        Q = self.W_q(x).view(B, L, H, D).transpose(1, 2)  # (B,H,L,D)
        K = self.W_k(x).view(B, L, H, D).transpose(1, 2)
        V = self.W_v(x).view(B, L, H, D).transpose(1, 2)

        # Number of top queries to keep
        u = max(1, int(self.factor * math.log(L + 1)))
        u = min(u, L)

        # Sample a reduced set of keys to estimate query importance
        sample_k = max(1, int(self.factor * math.log(L + 1)))
        sample_k = min(sample_k, L)

        idx_sample = torch.randint(0, L, (sample_k,))
        K_sample   = K[:, :, idx_sample, :]               # (B,H,sample_k,D)

        # QK score for each query against sampled keys
        QK_sample  = torch.einsum('bhld,bhsd->bhls', Q, K_sample) * self.scale
        # Sparsity measure: max - mean
        M          = QK_sample.max(dim=-1).values - QK_sample.mean(dim=-1)
        # Select top-u queries
        M_top_idx  = M.topk(u, dim=-1).indices                # (B,H,u)

        # Gather top queries
        Q_reduce   = Q.gather(2,
            M_top_idx.unsqueeze(-1).expand(B, H, u, D))       # (B,H,u,D)

        # Full attention for top queries only
        scores     = torch.einsum('bhud,bhld->bhul', Q_reduce, K) * self.scale
        attn       = torch.softmax(scores, dim=-1)             # (B,H,u,L)
        V_top      = torch.einsum('bhul,bhld->bhud', attn, V)  # (B,H,u,D)

        # Initialise output as mean of V (default for non-top queries)
        out = V.mean(dim=2, keepdim=True).expand(B, H, L, D).clone()

        # Write top-u results back
        out.scatter_(2,
            M_top_idx.unsqueeze(-1).expand(B, H, u, D),
            V_top)

        out = out.transpose(1, 2).reshape(B, L, H * D)
        return self.out(out)


class _InformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, factor):
        super().__init__()
        self.attn   = _ProbSparseAttention(d_model, n_heads, factor)
        self.ff     = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1  = nn.LayerNorm(d_model)
        self.norm2  = nn.LayerNorm(d_model)
        self.drop   = nn.Dropout(0.1)

        # Distilling conv: halves the sequence length after each layer
        self.distil = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.norm1(x + self.drop(self.attn(x)))
        x = self.norm2(x + self.drop(self.ff(x)))
        # Distil: (B, L, d) → transpose → conv → pool → transpose back
        x = self.distil(x.transpose(1, 2)).transpose(1, 2)
        return x


class _InformerNet(nn.Module):
    """
    Lightweight Informer for regime classification.

    (B, T, C) → embed → ProbSparse layers with distilling
              → global avg pool → classify
    """
    def __init__(self, n_features, seq_len, d_model, n_heads,
                 n_layers, d_ff, factor, n_classes):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)

        # Positional encoding (fixed sinusoidal)
        pe  = torch.zeros(seq_len, d_model)
        pos = torch.arange(seq_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, T, d)

        self.layers = nn.ModuleList([
            _InformerEncoderLayer(d_model, n_heads, d_ff, factor)
            for _ in range(n_layers)
        ])

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_classes)
        )

    def forward(self, x):
        # x: (B, T, C)
        x = self.input_proj(x) + self.pe[:, :x.size(1), :]
        for layer in self.layers:
            x = layer(x)
        x = x.mean(dim=1)         # global average pool
        return self.head(x)
    
    def get_embeddings(self, x):
        """Get embeddings before classification head."""
        x = self.input_proj(x) + self.pe[:, :x.size(1), :]
        for layer in self.layers:
            x = layer(x)
        return x.mean(dim=1)  # Return pooled embeddings


class _SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):  return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]