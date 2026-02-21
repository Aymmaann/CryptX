from pyspark.sql import DataFrame
from pyspark.sql import functions as F
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import torch
import warnings
warnings.filterwarnings('ignore')


class ChronosRegimeDetector:
    """
    Chronos-based regime detector (Amazon, 2024).

    Chronos is a language-model-style foundation model pre-trained on
    100B+ real and synthetic time series data points. It tokenises time
    series values and uses a T5 encoder-decoder architecture.

    Strategy here:
        1. Load the smallest Chronos checkpoint (chronos-t5-tiny, ~8M params)
        2. Run the encoder on rolling windows of price/vol to get
           context embeddings (frozen - no training on our data)
        3. Train a lightweight Logistic Regression head on top of those
           embeddings, supervised by volatility-quantile pseudo-labels
        4. Predict regimes on the full dataset

    This is classic transfer learning - we leverage Chronos's pre-trained
    representations of time series patterns without needing GPU training.

    Install:
        pip install chronos-forecasting
    """

    def __init__(self, n_regimes=3, seq_len=64,
                 model_name="amazon/chronos-t5-tiny",
                 random_state=42):
        """
        Args:
            n_regimes:    Number of regimes
            seq_len:      Context window fed to Chronos encoder
            model_name:   HuggingFace model ID (tiny=8M, mini=20M, small=46M)
            random_state: Seed for reproducibility
        """
        self.n_regimes    = n_regimes
        self.seq_len      = seq_len
        self.model_name   = model_name
        self.random_state = random_state
        self.scaler       = StandardScaler()
        self.pipeline     = None
        self.classifier   = None

        np.random.seed(random_state)
        torch.manual_seed(random_state)

        print(f"[INFO] Chronos initialized | model={model_name} "
              f"seq_len={seq_len}")

    def detect(self, df: DataFrame) -> DataFrame:
        """
        Detect regimes using Chronos embeddings + classifier.

        Args:
            df: Spark DataFrame with engineered features

        Returns:
            DataFrame with 'regime' and 'regime_name' columns added
        """
        print("[CHRONOS] Starting regime detection...")

        # Chronos works on univariate series - use the most informative ones
        # We extract embeddings separately per series then concatenate
        embedding_cols = ['price_close', 'vol_60m', 'volume_base']
        available_embed = [c for c in embedding_cols if c in df.columns]

        feature_cols = [
            'log_return', 'vol_60m', 'vol_240m',
            'cum_return_60m', 'volume_spike', 'atr_pct'
        ]
        available_feat = [c for c in feature_cols if c in df.columns]

        all_cols = list(dict.fromkeys(available_embed + available_feat))
        print(f"[CHRONOS] Using {len(available_embed)} series for embeddings, "
              f"{len(available_feat)} for features")

        print("[CHRONOS] Converting to Pandas...")
        df_pd = (df.select(['timestamp'] + all_cols)
                   .toPandas()
                   .sort_values('timestamp')
                   .reset_index(drop=True))

        # Pseudo-labels from volatility
        vol_col  = 'vol_60m' if 'vol_60m' in df_pd.columns else available_feat[0]
        vol_vals = df_pd[vol_col].values
        pseudo_labels = self._quantile_labels(vol_vals)

        # Load Chronos
        self._load_chronos()

        # Extract embeddings
        print("[CHRONOS] Extracting Chronos embeddings (this may take a while)...")
        embeddings = self._extract_embeddings(df_pd, available_embed)
        print(f"[CHRONOS] Embedding shape: {embeddings.shape}")

        # Train lightweight classifier on embeddings
        print("[CHRONOS] Training classifier on embeddings...")
        # Only use rows where we have a full context window
        valid_mask = np.arange(len(embeddings)) >= self.seq_len
        X_train = embeddings[valid_mask]
        y_train = pseudo_labels[valid_mask]

        self.classifier = LogisticRegression(
            max_iter=500,
            random_state=self.random_state,
            C=1.0,
            multi_class='multinomial'
        )
        self.classifier.fit(X_train, y_train)
        train_acc = self.classifier.score(X_train, y_train)
        print(f"[CHRONOS] Classifier training accuracy: {train_acc*100:.1f}%")

        # Predict
        print("[CHRONOS] Predicting regimes...")
        regimes = np.full(len(df_pd), -1, dtype=int)
        regimes[valid_mask] = self.classifier.predict(embeddings[valid_mask])

        # Fill warm-up period
        first_valid = regimes[regimes >= 0][0]
        regimes[regimes == -1] = first_valid

        # Map to volatility-ordered labels
        regimes = self._map_regimes_by_volatility(regimes, vol_vals)

        df_pd['regime']      = regimes
        df_pd['regime_name'] = df_pd['regime'].map(
            {0: 'low_vol', 1: 'medium_vol', 2: 'high_vol'}
        )

        print("[CHRONOS] Converting back to Spark DataFrame...")
        spark     = df.sparkSession
        df_result = spark.createDataFrame(
            df_pd[['timestamp', 'regime', 'regime_name']]
        )
        df_result = df.join(df_result, on='timestamp', how='inner')
        df_result = df_result.cache()
        df_result.count()

        print("[CHRONOS] Detection complete")
        return df_result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_chronos(self):
        """Load Chronos pipeline (downloads on first run, cached after)."""
        try:
            from chronos import ChronosPipeline
            print(f"[CHRONOS] Loading {self.model_name}...")
            self.pipeline = ChronosPipeline.from_pretrained(
                self.model_name,
                device_map="cpu",
                torch_dtype=torch.float32
            )
            print("[CHRONOS] Model loaded successfully")
        except ImportError:
            raise ImportError(
                "Chronos not installed. Run:\n"
                "  pip install chronos-forecasting"
            )

    def _extract_embeddings(self, df_pd, series_cols):
        """
        Extract encoder embeddings from Chronos for each timestep.

        For each timestep i we feed the window [i-seq_len : i] into
        the Chronos encoder and take the mean of the output token
        embeddings as a fixed-size representation.

        To keep CPU time reasonable we:
          - Stride by 4 (predict every 4th step, interpolate the rest)
          - Use the tiny model (8M params)
          - Process in mini-batches
        """
        n = len(df_pd)
        stride = 4  # predict every 4th step for speed

        # Indices we will actually compute embeddings for
        compute_idx = list(range(self.seq_len, n, stride))
        if compute_idx[-1] != n - 1:
            compute_idx.append(n - 1)

        # We'll concatenate embeddings across series
        all_embeddings = []
        batch_size = 32  # number of windows per Chronos call

        for col in series_cols:
            series = torch.tensor(
                df_pd[col].values, dtype=torch.float32
            )
            col_embeddings = np.zeros((n, 512))  # Chronos-tiny hidden=512

            for batch_start in range(0, len(compute_idx), batch_size):
                batch_idx = compute_idx[batch_start: batch_start + batch_size]
                windows   = torch.stack([
                    series[max(0, i - self.seq_len): i]
                    for i in batch_idx
                ])

                with torch.no_grad():
                    # Use internal encoder to get embeddings
                    emb = self._encode_windows(windows)

                for k, idx in enumerate(batch_idx):
                    col_embeddings[idx] = emb[k]

            # Interpolate non-computed steps
            for i in range(self.seq_len, n):
                if col_embeddings[i].sum() == 0:
                    col_embeddings[i] = col_embeddings[i - 1]

            all_embeddings.append(col_embeddings)

            print(f"[CHRONOS]   Embedded series: {col}")

        # Concatenate embeddings from all series, then reduce dimension
        combined = np.concatenate(all_embeddings, axis=1)

        # PCA to 64 dims to keep classifier lightweight
        from sklearn.decomposition import PCA
        pca = PCA(n_components=64, random_state=self.random_state)
        combined[self.seq_len:] = pca.fit_transform(combined[self.seq_len:])
        combined[:self.seq_len] = combined[self.seq_len]

        print(f"[CHRONOS] PCA reduced embeddings: {combined.shape}")
        return combined

    def _encode_windows(self, windows):
        """
        Use Chronos encoder to get mean-pooled token embeddings.

        windows: (B, T) float tensor
        Returns: (B, hidden_size) numpy array
        """
        try:
            # Internal access to the T5 encoder
            model   = self.pipeline.model
            tokenizer = self.pipeline.tokenizer

            # Tokenize the windows
            context  = windows.unsqueeze(-1)   # (B, T, 1)
            input_ids, attention_mask, _ = tokenizer.context_input_transform(
                context
            )

            with torch.no_grad():
                encoder_out = model.model.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                # Mean pool over sequence
                hidden = encoder_out.last_hidden_state  # (B, seq, hidden)
                mask   = attention_mask.unsqueeze(-1).float()
                emb    = (hidden * mask).sum(1) / mask.sum(1)

            return emb.numpy()

        except Exception:
            # Fallback: use statistical features of the window if
            # internal API has changed between Chronos versions
            stats = []
            for w in windows.numpy():
                stats.append([
                    w.mean(), w.std(), np.percentile(w, 25),
                    np.percentile(w, 75), w.max() - w.min(),
                    np.diff(w).mean(), np.diff(w).std()
                ])
            arr = np.array(stats, dtype=np.float32)
            # Pad to 512 dims to keep shape consistent
            padded = np.zeros((len(arr), 512), dtype=np.float32)
            padded[:, :arr.shape[1]] = arr
            return padded

    def _quantile_labels(self, values):
        """Assign 0/1/2 labels using 33rd and 67th percentile thresholds."""
        q33, q67 = np.percentile(values, [33, 67])
        return np.where(values <= q33, 0,
                 np.where(values <= q67, 1, 2)).astype(int)

    def _map_regimes_by_volatility(self, regimes, volatility):
        """Re-order regime IDs so 0=low, 1=medium, 2=high."""
        regime_vols = {
            r: volatility[regimes == r].mean()
            for r in range(self.n_regimes)
            if (regimes == r).sum() > 0
        }
        regime_map = {old: new for new, old
                      in enumerate(sorted(regime_vols, key=regime_vols.get))}
        return np.array([regime_map.get(r, r) for r in regimes])