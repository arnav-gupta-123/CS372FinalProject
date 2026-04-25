import numpy as np
from sklearn.metrics import accuracy_score

#speller to predict spelling based on models

class Speller:
    def __init__(self, speller_grid):
        self.speller_grid = speller_grid
        self.metrics = {}

    def run(self, pcr, clf, X, y):
        raw = pcr['raw_data']
        char_ch_idx = pcr['character_channels']
        target_indices = pcr['current_target_events'][:, 2].astype(int)
        n_chars = len(target_indices)

        # get classifier scores
        scores = clf.test(X, y) if clf is not None else X

        raw_data = raw.get_data(picks=char_ch_idx)
        event_samples = pcr['epochs'].events[:, 0].astype(int)

        valid = event_samples < raw_data.shape[1]
        event_samples = event_samples[valid]
        scores = scores[valid]

        n_epochs = len(event_samples)
        flashes_per_char = n_epochs // n_chars

        pred, targets = [], []

        # build prediction and target spelling arrangement. 
        for i in range(n_chars):
            start = i * flashes_per_char
            end = (i + 1) * flashes_per_char

            char_scores = np.zeros(len(char_ch_idx))

            for j in range(start, min(end, len(event_samples))):
                sample_idx = event_samples[j]
                if sample_idx >= raw_data.shape[1]:
                    continue
                flashed = raw_data[:, sample_idx] > 0.5
                char_scores[flashed] += scores[j]

            pred_char = self.speller_grid[np.argmax(char_scores)]
            true_char = self.speller_grid[target_indices[i] - 1] if target_indices[i] > 0 else '_'

            pred.append(pred_char)
            targets.append(true_char)

        # compare
        self.metrics['target'] = ''.join(targets)
        self.metrics['prediction'] = ''.join(pred)
        self.metrics['accuracy'] = accuracy_score(targets, pred)

        return pred

    def get_metrics(self):
        return self.metrics