# Experiment 9: Binary ASK and FSK Modulation & Detection

## 📑 Overview
This experiment explores the generation, transmission, and detection of binary digital modulation schemes, specifically **Binary Amplitude Shift Keying (BASK / OOK)** and **Binary Frequency Shift Keying (BFSK)**. It provides a deep dive into the differences between **coherent** and **noncoherent** reception, validates the orthogonality of FSK basis functions, and evaluates Bit Error Rate (BER) performance under Additive White Gaussian Noise (AWGN).

---

## 🎯 Objectives
1. Generate and visualize passband waveforms and power spectra for BASK and BFSK.
2. Implement and compare **correlator-based coherent detection** and **energy/envelope-based noncoherent detection**.
3. Validate the mathematical orthogonality of BFSK basis signals via inner product calculation.
4. Analyze the impact of tone spacing (orthogonal vs. non-orthogonal) on system performance.
5. Estimate and plot BER curves to compare theoretical and simulated performance across varying $E_b/N_0$ levels.

---

## 📚 Theoretical Background & Concepts Used

### 1. Binary ASK (OOK) and BFSK
* **BASK (On-Off Keying)**: Transmits a carrier for bit '1' and nothing for bit '0'. It is simple to generate but highly susceptible to noise and fading because the decision threshold must be dynamically adjusted based on signal power.
* **BFSK**: Transmits one frequency ($f_1$) for bit '0' and another ($f_2$) for bit '1'. It maintains a constant envelope, making it highly robust against non-linear amplifiers and fading.

### 2. Orthogonality in FSK
For the receiver to optimally distinguish between $f_1$ and $f_2$ without cross-interference, the basis signals must be **orthogonal** over the bit duration $T_b$. 
The condition for orthogonality is that the frequency separation $\Delta f = |f_2 - f_1|$ must be an integer multiple of $\frac{1}{2T_b}$. The minimum frequency separation for coherent detection is $\Delta f = \frac{1}{2T_b}$, and for noncoherent detection, it is $\Delta f = \frac{1}{T_b}$.

### 3. Coherent vs. Noncoherent Detection
* **Coherent Detection**: Requires the receiver to have exact knowledge of the carrier phase. It uses **correlators** (or matched filters) to project the received signal onto the known basis functions. It offers the best BER performance.
* **Noncoherent Detection**: Does not require phase synchronization. It uses **envelope detectors** (calculating the magnitude of the I/Q components). It is simpler to implement but suffers a performance penalty of roughly 1 to 2 dB compared to coherent detection.

### 4. Bit Error Rate (BER) Theory
When normalized by the *average* bit energy $E_b$:
* **Coherent BASK & Coherent BFSK**: $P_e = Q\left(\sqrt{\frac{E_b}{N_0}}\right)$
* **Noncoherent BFSK**: $P_e = \frac{1}{2} \exp\left(-\frac{E_b}{2N_0}\right)$

---

## 💻 How the Code Works

The Python script is heavily vectorized using NumPy to ensure fast execution (under a second in Google Colab) and is structured into sequential experiments.

### Step-by-Step Workflow:
1. **Strict Normalization**: All basis functions ($\phi(t)$) are mathematically normalized to have an energy of exactly 1.0. This eliminates scaling ambiguities and allows the use of exact theoretical decision thresholds (e.g., $\sqrt{2}/2$ for BASK).
2. **Signal Generation**: Generates random bit sequences and maps them to passband signals using vectorized operations (`np.repeat`, `np.where`) instead of slow `for` loops.
3. **Channel Modeling**: Adds AWGN noise scaled precisely to the desired $E_b/N_0$ ratio.
4. **Detection Logic**:
   * *Coherent*: Computes the inner product (correlation) of the received signal with the basis functions.
   * *Noncoherent*: Computes the In-phase (I) and Quadrature (Q) correlations, then calculates the envelope $\sqrt{I^2 + Q^2}$.
5. **Parameter Sweeps**: Iterates through an $E_b/N_0$ range (0 to 12 dB) to build the BER curves, comparing simulated results against theoretical $Q$-function and exponential curves.

---

## ⚠️ Discrepancies, Reasons, and Mitigations

During the development and debugging of this simulation, several critical discrepancies were identified and resolved.

### Discrepancy 1: 100% Bit Error Rate (Inverted Decision Logic)
* **Observation**: Initial simulations yielded a BER of ~0.5 (random guessing) or 1.0 (100% errors) for BFSK, regardless of SNR.
* **Reason**: The mapping assigned bit '0' to $f_1$ and bit '1' to $f_2$. However, the decision rule was incorrectly coded as `if Z1 > Z2: decide 1`. This inverted the output bits.
* **Mitigation**: Corrected the decision logic to `D = Z2 - Z1`. If $D > 0$, $f_2$ has more energy, correctly mapping to bit '1'.

### Discrepancy 2: BASK and BFSK BER Curves Not Overlapping
* **Observation**: Theory states Coherent BASK and Coherent BFSK should have the exact same BER curve when plotted against *average* $E_b/N_0$. The initial simulation showed BASK performing 3 dB worse.
* **Reason**: In BASK, bit '1' has energy $E_{peak}$ and bit '0' has 0 energy. The *average* energy is $E_{peak}/2$. The code was initially normalizing the peak energy to 1, making the average energy 0.5, which artificially shifted the BASK curve by 3 dB.
* **Mitigation**: Scaled the BASK amplitude by $\sqrt{4/T_b}$ instead of $\sqrt{2/T_b}$. This ensures the peak energy is 2, making the *average* energy exactly 1.0, perfectly aligning the simulated curve with the theoretical $Q(\sqrt{E_b/N_0})$ curve.

### Discrepancy 3: Matplotlib Dimension Mismatch Error
* **Observation**: The code crashed with `ValueError: x and y must have same first dimension` when plotting the passband waveforms.
* **Reason**: The time vector `t_bit` was defined for exactly *one* bit (50 samples), but the code attempted to plot it against a signal spanning *three* bits (150 samples).
* **Mitigation**: Created a dedicated visualization time vector `t_vis = np.arange(N_bits_vis * spb) / Fs` that perfectly matches the length of the multi-bit signal arrays.

### Discrepancy 4: `AttributeError: module 'numpy' has no attribute 'erfc'`
* **Observation**: The script crashed immediately at the theoretical BER calculation step.
* **Reason**: The complementary error function `erfc` is not a part of the core `numpy` library; it resides in `scipy.special`.
* **Mitigation**: Added `from scipy.special import erfc` to the imports and updated the `qfunc` definition.

---

## 🚀 How to Run

1. Open **Google Colab** (or any local Jupyter Notebook environment).
2. Ensure `numpy`, `matplotlib`, and `scipy` are available (they are pre-installed in Colab).
3. Copy and paste the entire Python script into a **single code cell**.
4. Run the cell. The script will print the theoretical validations, observations, and render all required matplotlib figures inline.

---

## 📊 Expected Outputs

Running the code will generate the following visualizations and console outputs:
1. **Passband Waveforms & Spectra**: Time-domain plots showing the amplitude shifts of BASK and frequency shifts of BFSK, alongside their Power Spectral Density (PSD) showing the single-lobe and dual-lobe characteristics.
2. **Mandatory Validation Printout**: Console output proving the inner product of orthogonal FSK signals is $\approx 0$, while non-orthogonal signals yield a non-zero cross-correlation.
3. **Correlator Outputs & Histograms**: A stem plot showing the distinct correlator outputs for '0' and '1', and a histogram showing two well-separated Gaussian distributions for the decision statistic $D$.
4. **BER Performance Curves**: A comprehensive log-scale plot comparing Simulated vs. Theoretical BER for Coherent/Noncoherent BASK and Orthogonal/Non-orthogonal BFSK, clearly demonstrating the 1-2 dB penalty of noncoherent detection and the penalty of non-orthogonal tone spacing.
