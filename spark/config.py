"""
Configuration file for Crypto Volatility Analysis Framework
Centralizes all paths, parameters, and settings for reproducibility.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class PathConfig:
    """File system paths configuration."""
    BASE_PATH: Path = Path(__file__).parent.parent 
    
    # Data directories
    RAW_DATA_PATH: Path = BASE_PATH / "data" / "raw"
    PROCESSED_DATA_PATH: Path = BASE_PATH / "data" / "processed"
    CHECKPOINT_PATH: Path = BASE_PATH / "data" / "checkpoints"
    
    # Spark directory
    SPARK_PATH: Path = BASE_PATH / "spark"
    
    # Output directories
    MODELS_PATH: Path = BASE_PATH / "models"
    RESULTS_PATH: Path = BASE_PATH / "results"
    LOGS_PATH: Path = BASE_PATH / "logs"
    
    def create_directories(self):
        """Create all necessary directories."""
        for path_attr in dir(self):
            if path_attr.endswith('_PATH') and not path_attr.startswith('_'):
                path = getattr(self, path_attr)
                path.mkdir(parents=True, exist_ok=True)


@dataclass
class DataIngestionConfig:
    """Data ingestion parameters."""
    # Binance API configuration
    BINANCE_API_BASE_URL: str = "https://api.binance.com"
    BINANCE_WS_BASE_URL: str = "wss://stream.binance.com:9443"
    
    # Data source parameters
    DEFAULT_SYMBOL: str = "BTCUSDT"
    DEFAULT_INTERVAL: str = "1m"
    
    # Historical data parameters
    HISTORICAL_START_MONTH: str = "2025-07"
    HISTORICAL_END_MONTH: str = "2025-12"
    
    # Streaming parameters
    WEBSOCKET_BUFFER_SIZE: int = 10000
    STREAMING_BATCH_SIZE: int = 1000
    STREAMING_PROCESS_INTERVAL: int = 60  # seconds
    
    # API rate limits
    REST_API_TIMEOUT: int = 10  # seconds
    REST_API_MAX_LIMIT: int = 1000  # max candles per request


@dataclass
class SparkConfig:
    """Spark configuration parameters."""
    APP_NAME: str = "CryptoVolatilityFramework"
    
    # Memory settings
    DRIVER_MEMORY: str = "4g"
    EXECUTOR_MEMORY: str = "4g"
    
    # Performance tuning
    SHUFFLE_PARTITIONS: int = 200
    MAX_PARTITION_BYTES: str = "128MB"
    ADAPTIVE_ENABLED: bool = True
    COALESCE_PARTITIONS_ENABLED: bool = True
    
    # Storage format
    DEFAULT_OUTPUT_FORMAT: str = "parquet"
    PARTITION_COLUMNS: List[str] = None
    
    def __post_init__(self):
        if self.PARTITION_COLUMNS is None:
            self.PARTITION_COLUMNS = ["year", "month"]


@dataclass
class DataQualityConfig:
    """Data quality validation thresholds."""
    # Price validation
    MIN_PRICE: float = 0.0
    MAX_PRICE_CHANGE_PERCENT: float = 50.0  # Max % change between candles
    
    # Volume validation
    MIN_VOLUME: float = 0.0
    
    # Temporal validation
    EXPECTED_TIME_GAP_SECONDS: int = 60  # for 1-minute candles
    MAX_ALLOWED_GAPS: int = 10  # Maximum acceptable gaps in data
    
    # Missing data tolerance
    MAX_MISSING_PERCENT: float = 1.0  # Max % of missing values allowed


@dataclass
class FeatureEngineeringConfig:
    """Configuration for Module 2 - Feature Engineering."""
    # Rolling window sizes (in minutes for 1-minute data)
    VOLATILITY_WINDOWS: List[int] = None
    VOLUME_WINDOWS: List[int] = None
    PRICE_WINDOWS: List[int] = None
    
    # Feature flags
    INCLUDE_ONCHAIN_FEATURES: bool = False
    INCLUDE_TECHNICAL_INDICATORS: bool = True
    INCLUDE_MICROSTRUCTURE_FEATURES: bool = True
    
    def __post_init__(self):
        if self.VOLATILITY_WINDOWS is None:
            self.VOLATILITY_WINDOWS = [5, 15, 30, 60, 240, 1440]  # 5min to 1day
        if self.VOLUME_WINDOWS is None:
            self.VOLUME_WINDOWS = [15, 60, 240, 1440]
        if self.PRICE_WINDOWS is None:
            self.PRICE_WINDOWS = [5, 15, 30, 60]


@dataclass
class RegimeDetectionConfig:
    """Configuration for Module 3 - Regime Detection."""
    # HMM parameters
    HMM_N_STATES: int = 3  # Low, Medium, High volatility
    HMM_N_ITERATIONS: int = 100
    HMM_TOLERANCE: float = 1e-4
    
    # Change point detection
    CHANGEPOINT_MIN_SIZE: int = 60  # Minimum regime duration (minutes)
    CHANGEPOINT_PENALTY: float = 1.0
    
    # Features for regime detection
    REGIME_FEATURES: List[str] = None
    
    def __post_init__(self):
        if self.REGIME_FEATURES is None:
            self.REGIME_FEATURES = [
                "log_return",
                "realized_volatility",
                "volume_base"
            ]


@dataclass
class VolatilityModelingConfig:
    """Configuration for volatility modeling."""
    # GARCH parameters
    GARCH_P: int = 1  # GARCH order
    GARCH_Q: int = 1  # ARCH order
    
    # Model types to compare
    MODEL_TYPES: List[str] = None
    
    # Train/test split
    TRAIN_SPLIT_RATIO: float = 0.8
    
    # Forecasting
    FORECAST_HORIZON: int = 60  # minutes ahead
    
    def __post_init__(self):
        if self.MODEL_TYPES is None:
            self.MODEL_TYPES = [
                "GARCH",
                "EGARCH",
                "GJR-GARCH",
                "Regime-GARCH"
            ]


@dataclass
class EvaluationConfig:
    """Model evaluation configuration."""
    # Metrics to compute
    METRICS: List[str] = None
    
    # Cross-validation
    N_FOLDS: int = 5
    
    # Robustness tests
    BOOTSTRAP_ITERATIONS: int = 1000
    
    def __post_init__(self):
        if self.METRICS is None:
            self.METRICS = [
                "RMSE",
                "MAE",
                "MAPE",
                "Log-Likelihood",
                "Directional Accuracy"
            ]


class Config:
    """Master configuration class combining all configs."""
    
    def __init__(self):
        self.paths = PathConfig()
        self.ingestion = DataIngestionConfig()
        self.spark = SparkConfig()
        self.data_quality = DataQualityConfig()
        self.feature_engineering = FeatureEngineeringConfig()
        self.regime_detection = RegimeDetectionConfig()
        self.volatility_modeling = VolatilityModelingConfig()
        self.evaluation = EvaluationConfig()
    
    def setup_environment(self):
        """Initialize the environment by creating necessary directories."""
        print("[INFO] Setting up project environment...")
        self.paths.create_directories()
        print("[INFO] All directories created successfully")
    
    def summary(self):
        """Print configuration summary."""
        print("\n" + "="*70)
        print("CRYPTO VOLATILITY ANALYSIS FRAMEWORK - CONFIGURATION SUMMARY")
        print("="*70)
        
        print(f"\n[PATHS]")
        print(f"  Base Path: {self.paths.BASE_PATH}")
        print(f"  Raw Data: {self.paths.RAW_DATA_PATH}")
        print(f"  Processed Data: {self.paths.PROCESSED_DATA_PATH}")
        
        print(f"\n[DATA INGESTION]")
        print(f"  Symbol: {self.ingestion.DEFAULT_SYMBOL}")
        print(f"  Interval: {self.ingestion.DEFAULT_INTERVAL}")
        print(f"  Historical Period: {self.ingestion.HISTORICAL_START_MONTH} to {self.ingestion.HISTORICAL_END_MONTH}")
        
        print(f"\n[SPARK]")
        print(f"  Driver Memory: {self.spark.DRIVER_MEMORY}")
        print(f"  Shuffle Partitions: {self.spark.SHUFFLE_PARTITIONS}")
        print(f"  Output Format: {self.spark.DEFAULT_OUTPUT_FORMAT}")
        
        print(f"\n[FEATURE ENGINEERING]")
        print(f"  Volatility Windows: {self.feature_engineering.VOLATILITY_WINDOWS}")
        print(f"  Include Technical Indicators: {self.feature_engineering.INCLUDE_TECHNICAL_INDICATORS}")
        
        print(f"\n[REGIME DETECTION]")
        print(f"  HMM States: {self.regime_detection.HMM_N_STATES}")
        print(f"  Features: {self.regime_detection.REGIME_FEATURES}")
        
        print(f"\n[VOLATILITY MODELING]")
        print(f"  GARCH Order: ({self.volatility_modeling.GARCH_P}, {self.volatility_modeling.GARCH_Q})")
        print(f"  Model Types: {self.volatility_modeling.MODEL_TYPES}")
        print(f"  Forecast Horizon: {self.volatility_modeling.FORECAST_HORIZON} minutes")
        
        print("\n" + "="*70 + "\n")


# Singleton instance
config = Config()


if __name__ == "__main__":
    # Display configuration and setup environment
    config.summary()
    config.setup_environment()