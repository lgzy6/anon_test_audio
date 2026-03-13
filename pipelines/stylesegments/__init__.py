"""Style Segments Pipeline - 统一的特征提取和风格分割管线"""

from .feature_extraction import FeatureExtractor, run_feature_extraction
from .speaker_embeddings import extract_speaker_embeddings
from .eta_projection import compute_eta_projection
from .phone_precompute import precompute_phones
from .style_extractor import SegmentStyleExtractor, build_style_extractor
from .phone_clusters import build_phone_clusters

__all__ = [
    'FeatureExtractor',
    'run_feature_extraction',
    'extract_speaker_embeddings',
    'compute_eta_projection',
    'precompute_phones',
    'SegmentStyleExtractor',
    'build_style_extractor',
    'build_phone_clusters',
]
