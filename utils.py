# Perform inference using the provided script
def perform_inference(model, test_set):
    si_snr_metric = ScaleInvariantSignalNoiseRatio()
    si_sdr_metric = ScaleInvariantSignalDistortionRatio()
    si_snr_values = []
    si_sdr_values = []
    for mix_path, s1_path, s2_path in test_set:
        est_sources = model.separate_file(path=mix_path)
        s1, _ = torchaudio.load(s1_path)
        s2, _ = torchaudio.load(s2_path)

        # Calculate SI-SNR and SI-SDR for original sources
        original_sources = torch.stack((s1, s2))
        si_snr_orig = si_snr_metric(original_sources, original_sources)
        si_sdr_orig = si_sdr_metric(original_sources, original_sources)

        # Calculate SI-SNR and SI-SDR for estimated sources
        si_snr_est = si_snr_metric(est_sources, original_sources)
        si_sdr_est = si_sdr_metric(est_sources, original_sources)

        # Calculate SI-SNR and SI-SDR improvement
        si_snr_improvement = si_snr_est - si_snr_orig
        si_sdr_improvement = si_sdr_est - si_sdr_orig

        # Append metric values to lists
        si_snr_values.append(si_snr_improvement.item())
        si_sdr_values.append(si_sdr_improvement.item())

    # Calculate average values
    avg_si_snr = sum(si_snr_values) / len(si_snr_values)
    avg_si_sdr = sum(si_sdr_values) / len(si_sdr_values)

    # Log average values
    print(f"Average SI-SNR Improvement: {avg_si_snr}")
    print(f"Average SI-SDR Improvement: {avg_si_sdr}")

    # Optionally, you can also return the average values if needed
    return avg_si_snr, avg_si_sdr


# Load dataset as a tuple (mix, s1, s2) (only paths)
def load_dataset(mix_folder, s1_folder, s2_folder):
    mix_files = os.listdir(mix_folder)
    dataset = []
    for mix_file in mix_files:
        mix_path = os.path.join(mix_folder, mix_file)
        s1_path = os.path.join(s1_folder, mix_file)
        s2_path = os.path.join(s2_folder, mix_file)
        dataset.append((mix_path, s1_path, s2_path))
    return dataset

# Perform 70:30 split
def train_test_split(dataset, split_ratio=0.7):
    random.shuffle(dataset)
    split_idx = int(len(dataset) * split_ratio)
    train_set = dataset[:split_idx]
    test_set = dataset[split_idx:]
    return train_set, test_set
