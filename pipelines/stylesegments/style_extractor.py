"""段级风格提取器"""

import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import normalize


class SegmentStyleExtractor:
    """段级风格提取器"""

    def __init__(self, pca_dim=64, min_segment_frames=3):
        self.pca_dim = pca_dim
        self.min_segment_frames = min_segment_frames
        self.pca = None
        self.fitted = False

    def _find_phone_segments(self, phones):
        if len(phones) == 0:
            return []
        boundaries = np.concatenate([[0], np.where(np.diff(phones) != 0)[0] + 1, [len(phones)]])
        segments = []
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i + 1]
            phone_id = phones[start]
            if phone_id != 0 and (end - start) >= self.min_segment_frames:
                segments.append((start, end, int(phone_id)))
        return segments

    def _segment_to_raw_embedding(self, frames):
        mu = frames.mean(axis=0)
        sigma = frames.std(axis=0) + 1e-8
        if len(frames) > 1:
            delta_mu = np.diff(frames, axis=0).mean(axis=0)
        else:
            delta_mu = np.zeros_like(mu)
        return np.concatenate([mu, sigma, delta_mu])

    def fit_incremental(self, eta_iterator, batch_size=1000, pca_dim=64):
        """增量拟合"""
        self.pca = IncrementalPCA(n_components=pca_dim, batch_size=batch_size)
        batch, total_segs = [], 0

        for eta_utt, phones_utt, _ in tqdm(eta_iterator, desc="Fitting PCA"):
            segments = self._find_phone_segments(phones_utt)
            for start, end, _ in segments:
                raw_emb = self._segment_to_raw_embedding(eta_utt[start:end])
                batch.append(normalize(raw_emb.reshape(1, -1))[0])
                total_segs += 1
                if len(batch) >= batch_size:
                    self.pca.partial_fit(np.array(batch))
                    batch = []

        if batch:
            self.pca.partial_fit(np.array(batch))

        self.fitted = True
        print(f"Fitted on {total_segs} segments, PCA dim: {pca_dim}")
        return self

    def extract_segment_styles(self, eta_utt, phones_utt):
        segments = self._find_phone_segments(phones_utt)
        results = []
        for start, end, phone_id in segments:
            raw = self._segment_to_raw_embedding(eta_utt[start:end])
            raw_norm = normalize(raw.reshape(1, -1))
            emb_pca = self.pca.transform(raw_norm)[0]
            results.append({
                'start': start, 'end': end, 'phone': phone_id,
                'style_emb': emb_pca, 'n_frames': end - start
            })
        return results

    def extract_utterance_style(self, eta_utt, phones_utt):
        seg_results = self.extract_segment_styles(eta_utt, phones_utt)
        if not seg_results:
            return np.zeros(self.pca.n_components_)
        embs = np.stack([s['style_emb'] for s in seg_results])
        weights = np.array([s['n_frames'] for s in seg_results], dtype=float)
        weights /= weights.sum()
        return (embs * weights[:, None]).sum(axis=0)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'pca': self.pca, 'pca_dim': self.pca_dim,
                'min_segment_frames': self.min_segment_frames, 'fitted': self.fitted
            }, f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        obj = cls(pca_dim=data['pca_dim'], min_segment_frames=data['min_segment_frames'])
        obj.pca = data['pca']
        obj.fitted = data['fitted']
        return obj


def build_style_extractor(eta_generator, ckpt_dir, pca_dim=64):
    """构建风格提取器"""
    extractor = SegmentStyleExtractor(pca_dim=pca_dim)
    extractor.fit_incremental(eta_generator, batch_size=1000, pca_dim=pca_dim)
    ckpt_dir = Path(ckpt_dir)
    extractor.save(ckpt_dir / 'style_extractor.pkl')
    print(f"Saved: {ckpt_dir / 'style_extractor.pkl'}")
    return extractor
