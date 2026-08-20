import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import pywt
import math
from numba import jit
import warnings



class MovingWindowDetection:
    """
    Moving Window Method for detecting shock waves in traffic data.
    
    This method uses a simple moving window to detect abrupt changes in speed
    or other traffic variables that indicate shock waves.
    """
    
    def __init__(self, window_size=10, threshold=1.5, min_distance=5):
        """
        Initialize the Moving Window detector with more sensitive defaults.
        
        Args:
            window_size: Size of the moving window in data points (smaller value = more sensitive)
            threshold: Threshold for detecting significant changes (smaller value = more sensitive)
            min_distance: Minimum distance between detected breakpoints
        """
        self.window_size = window_size
        self.threshold = threshold
        self.min_distance = min_distance
        
    def detect(self, time, data):
        """
        Detect shock waves using the moving window method.
        
        Args:
            time: Array of time points
            data: Array of data (speed, position, etc.)
            
        Returns:
            breakpoints: Indices where shock waves are detected
            diagnostics: Additional information about the detection
        """
        n = len(data)
        if n < 2 * self.window_size:
            return [], {"total_cost": 0, "num_segments": 1, "avg_segment_length": n}
        
        # Calculate moving standard deviation
        std_dev = np.zeros(n)
        for i in range(self.window_size, n - self.window_size):
            window_data = data[i-self.window_size:i+self.window_size]
            std_dev[i] = np.std(window_data)
        
        # Detect points where standard deviation exceeds threshold
        mean_std = np.mean(std_dev[std_dev > 0])
        if mean_std == 0:  # Handle case where all std_dev values are 0
            mean_std = 1.0
        threshold_value = mean_std * self.threshold
        
        # Find potential breakpoints
        potential_breakpoints = []
        for i in range(self.window_size, n - self.window_size):
            if std_dev[i] > threshold_value:
                # Check if this is a local maximum of standard deviation
                if (std_dev[i] > std_dev[i-1] and std_dev[i] > std_dev[i+1]):
                    potential_breakpoints.append(i)
        
        # Ensure minimum distance between breakpoints
        breakpoints = []
        if potential_breakpoints:
            breakpoints = [potential_breakpoints[0]]
            for bp in potential_breakpoints[1:]:
                if bp - breakpoints[-1] >= self.min_distance:
                    breakpoints.append(bp)
        
        # Compute diagnostics
        diagnostics = {
            "total_cost": np.sum(std_dev),
            "num_segments": len(breakpoints) + 1,
            "avg_segment_length": n / (len(breakpoints) + 1),
            "std_dev": std_dev,
            "threshold": threshold_value
        }
        
        # If no breakpoints were found, adjust parameters and try again
        if not breakpoints and self.threshold > 1.0:
            # Reduce threshold and try again
            backup_threshold = self.threshold
            self.threshold = max(0.8, self.threshold * 0.6)  # Reduce threshold by 40%
            print(f"No breakpoints found with threshold {backup_threshold}. Retrying with {self.threshold}...")
            breakpoints, diagnostics = self.detect(time, data)
            self.threshold = backup_threshold  # Reset threshold for future calls
        
        print(f"Moving Window detected {len(breakpoints)} breakpoints")
        return breakpoints, diagnostics


class WaveletTransformDetection:
    """
    Wavelet Transform Method for detecting shock waves in traffic data.
    
    This method uses wavelet analysis to detect and locate shock waves in traffic flow.
    Enhanced to detect more change points by using lower threshold and smaller min_distance.
    """
    
    def __init__(self, wavelet='mexh', max_scale=32, threshold_factor=1.2, min_distance=1):
        """
        Initialize the Wavelet Transform detector with more sensitive parameters.
        
        Args:
            wavelet: Wavelet to use ('mexh' for Mexican hat wavelet is good for peaks)
            max_scale: Maximum scale for wavelet analysis (smaller value = more sensitive to local changes)
            threshold_factor: Factor to determine energy threshold (smaller value = more change points)
            min_distance: Minimum distance between detected shock waves (smaller = more change points)
        """
        self.wavelet = wavelet
        self.max_scale = max_scale
        self.threshold_factor = threshold_factor
        self.min_distance = min_distance
        
    def detect(self, time, data):
        """
        Detect shock waves using wavelet transform.
        
        Args:
            time: Array of time points
            data: Array of data (speed, position, etc.)
            
        Returns:
            breakpoints: Indices where shock waves are detected
            diagnostics: Additional information about the detection
        """
        n = len(data)
        if n < 2 * self.min_distance:
            return [], {"total_cost": 0, "num_segments": 1, "avg_segment_length": n}
        
        # Normalize the data
        normalized_data = (data - np.mean(data)) / np.std(data)
        
        # Apply the continuous wavelet transform
        # Use a smaller subset of scales focused on typical shock wave frequencies
        scales = np.arange(1, self.max_scale + 1)
        coeffs, freqs = pywt.cwt(normalized_data, scales, self.wavelet)
        
        # Calculate wavelet-based energy
        energy = np.mean(np.abs(coeffs)**2, axis=0)
        
        # Set threshold based on energy statistics
        energy_mean = np.mean(energy)
        energy_std = np.std(energy)
        threshold = energy_mean + self.threshold_factor * energy_std
        
        # If standard deviation is very small, set a minimum threshold
        if energy_std < 0.01 * energy_mean:
            threshold = energy_mean * 1.1  # Just 10% above mean
        
        # Find potential breakpoints (energy peaks above threshold)
        potential_breakpoints = []
        for i in range(1, n-1):
            if energy[i] > threshold and energy[i] > energy[i-1] and energy[i] > energy[i+1]:
                potential_breakpoints.append(i)
        
        # If few breakpoints are found, try a lower threshold
        if len(potential_breakpoints) < 3:
            lower_threshold = energy_mean + 0.5 * energy_std  # Much lower threshold
            for i in range(1, n-1):
                if (energy[i] > lower_threshold and energy[i] > energy[i-1] and 
                    energy[i] > energy[i+1] and i not in potential_breakpoints):
                    potential_breakpoints.append(i)
        
        # Ensure minimum distance between breakpoints
        breakpoints = []
        if potential_breakpoints:
            # Sort breakpoints by energy value (highest first)
            sorted_breakpoints = sorted(
                [(i, energy[i]) for i in potential_breakpoints],
                key=lambda x: x[1], reverse=True
            )
            
            # Take the highest energy breakpoints first
            selected_points = set()
            for idx, _ in sorted_breakpoints:
                if all(abs(idx - prev) >= self.min_distance for prev in selected_points):
                    selected_points.add(idx)
            
            # Convert back to sorted list
            breakpoints = sorted(list(selected_points))
        
        # Compute diagnostics
        diagnostics = {
            "energy": energy,
            "threshold": threshold,
            "num_segments": len(breakpoints) + 1,
            "avg_segment_length": n / (len(breakpoints) + 1),
            "max_energy": np.max(energy),
            "min_energy": np.min(energy)
        }
        
        print(f"Wavelet Transform detected {len(breakpoints)} breakpoints")
        return breakpoints, diagnostics


class CUSUMDPDetection:
    """
    CUSUM-DP Method for detecting shock waves in traffic data.
    
    This hybrid approach combines CUSUM for candidate generation with dynamic 
    programming for optimal selection of change points.
    """
    
    def __init__(self, cusum_threshold=7, cusum_drift=1.0, dp_penalty=100, 
                 min_cusum_length=10, min_segment_length=10, max_candidates=None):
        """
        Initialize the CUSUM-DP detector.
        
        Args:
            cusum_threshold: Detection threshold for CUSUM
            cusum_drift: Drift parameter for CUSUM
            dp_penalty: Penalty value for adding a changepoint in DP
            min_cusum_length: Minimum segment length for CUSUM
            min_segment_length: Minimum segment length for final segments
            max_candidates: Maximum number of candidate change points
        """
        self.cusum_threshold = cusum_threshold
        self.cusum_drift = cusum_drift
        self.dp_penalty = dp_penalty
        self.min_cusum_length = min_cusum_length
        self.min_segment_length = min_segment_length
        self.max_candidates = max_candidates
    
    def detect(self, data):
        """
        Detect shock waves using the CUSUM-DP method.
        
        Args:
            data: Array of data (time, distance, velocity, etc.) 
            

            'time': result['time'][dp],
            'distance': result['distance'][dp],
            'velocity': result['velocity'][dp],

        Returns:
            breakpoints: Indices where shock waves are detected
            diagnostics: Additional information about the detection
        """
        # Step 1: Generate candidate change points using CUSUM
        candidates, s_pos, s_neg = self._generate_cusum_candidates(
            data['velocity'], 
            cusum_threshold=self.cusum_threshold,
            cusum_drift=self.cusum_drift,
            min_segment_length=self.min_cusum_length,
            max_candidates=self.max_candidates
        )
        
        # Step 2: Use dynamic programming to find optimal subset of candidates
        optimal_changepoints = self._select_optimal_changepoints(
            data['time'], data['distance'], candidates,
            dp_penalty=self.dp_penalty,
            min_segment_length=self.min_segment_length
        )
        
        # Calculate slope and other diagnostics for each detected segment
        segments = []
        if optimal_changepoints:
            start_idx = 0
            
            for bp in optimal_changepoints + [len(data)-1]:
                segment_time = data['time'][start_idx:bp+1]
                segment_data = data['distance'][start_idx:bp+1]
                
                if len(segment_time) > 1:
                    # Calculate slope using linear regression
                    coeffs = np.polyfit(segment_time, segment_data, 1)
                    slope, intercept = coeffs
                    
                    # Calculate metrics
                    y_pred = slope * segment_time + intercept
                    residuals = segment_data - y_pred
                    mse = np.mean(residuals ** 2)
                    r2 = 1 - np.sum(residuals ** 2) / np.sum((segment_data - np.mean(segment_data)) ** 2)
                    
                    segments.append({
                        'start_idx': start_idx,
                        'end_idx': bp,
                        'slope': slope,
                        'intercept': intercept,
                        'mse': mse,
                        'r2': r2,
                        'duration': segment_time[-1] - segment_time[0]
                    })
                
                start_idx = bp + 1
        
        # Prepare diagnostics
        diagnostics = {
            "cusum": {
                "s_pos": s_pos,
                "s_neg": s_neg,
                "candidates": candidates
            },
            "num_segments": len(optimal_changepoints) + 1,
            "avg_segment_length": len(data) / (len(optimal_changepoints) + 1),
            "segments": segments
        }
        
        return optimal_changepoints, diagnostics
    
    def _generate_cusum_candidates(self, velocity, cusum_threshold=7, cusum_drift=1.0, 
                                   min_segment_length=10, max_candidates=None):
        """
        Generate candidate change points using CUSUM on velocity.
        
        Args:
            velocity: Velocity Data values
            cusum_threshold: Detection threshold for CUSUM
            cusum_drift: Drift parameter for CUSUM
            min_segment_length: Minimum segment length
            max_candidates: Maximum number of candidate change points
            
        Returns:
            List of candidate changepoint indices, positive and negative CUSUM values
        """
        n = len(velocity)
        if n < 2 * min_segment_length:
            return [], np.zeros(n), np.zeros(n)
        
        # Initialize CUSUM statistics
        S_pos = np.zeros(n)
        S_neg = np.zeros(n)
        
        # Compute mean for initial segment
        mean_velocity = np.mean(velocity[:min_segment_length])
        
        last_cp = 0
        candidates = []
        
        # Compute CUSUM statistics on velocity
        for i in range(min_segment_length, n):
            # Deviation from target
            deviation = velocity[i] - mean_velocity
            
            # Update positive and negative CUSUMs
            S_pos[i] = max(0, S_pos[i-1] + deviation - cusum_drift)
            S_neg[i] = max(0, S_neg[i-1] - deviation - cusum_drift)
            
            # Check if either CUSUM exceeds threshold
            if (S_pos[i] > cusum_threshold or S_neg[i] > cusum_threshold) and i - last_cp >= min_segment_length:
                candidates.append(i)
                
                # Reset CUSUM statistics
                S_pos[i] = 0
                S_neg[i] = 0
                
                # Update mean for next segment
                if i + min_segment_length < n:
                    mean_velocity = np.mean(velocity[i:i+min_segment_length])
                else:
                    mean_velocity = np.mean(velocity[i:])
                
                last_cp = i
        
        # Limit candidates if needed
        if max_candidates is not None and len(candidates) > max_candidates:
            strengths = [max(S_pos[c], S_neg[c]) for c in candidates]
            strongest_indices = np.argsort(strengths)[-max_candidates:]
            candidates = [candidates[i] for i in strongest_indices]
        
        return candidates, S_pos, S_neg
    
    def _select_optimal_changepoints(self, x, y, candidates, dp_penalty=100, min_segment_length=10):
        """
        Use dynamic programming to select optimal subset of candidate change points.
        
        Args:
            x: Time points
            y: Position values
            candidates: List of candidate change points
            dp_penalty: Penalty value for adding a changepoint in DP
            min_segment_length: Minimum segment length
            
        Returns:
            List of optimal changepoint indices
        """
        n = len(y)
        
        # Add beginning and end points to create complete segments
        all_points = [0] + sorted(candidates) + [n-1]
        
        # Initialize DP arrays
        m = len(all_points)
        dp = np.full(m, np.inf)  # Minimum cost up to point i
        prev = np.full(m, -1, dtype=int)  # Previous breakpoint for reconstruction
        
        # Precompute segment costs for all candidate segments
        segment_costs = {}
        for i in range(m-1):
            for j in range(i+1, m):
                start, end = all_points[i], all_points[j]
                if end - start >= min_segment_length:
                    segment_costs[(i, j)] = self._calculate_segment_cost(x[start:end+1], y[start:end+1])
        
        # Base case
        dp[0] = 0
        
        # Dynamic Programming over candidate points
        for j in range(1, m):
            for i in range(j):
                start, end = all_points[i], all_points[j]
                if end - start < min_segment_length:
                    continue
                    
                cost = dp[i] + segment_costs[(i, j)] + (dp_penalty if i > 0 else 0)
                if cost < dp[j]:
                    dp[j] = cost
                    prev[j] = i
        
        # Reconstruct solution
        optimal_indices = []
        curr = m - 1
        while curr > 0:
            prev_idx = prev[curr]
            if prev_idx > 0:  # Don't include the starting point (0)
                optimal_indices.append(all_points[prev_idx])
            curr = prev_idx
        
        # Return sorted optimal change points
        return sorted(optimal_indices)
    
    def _calculate_segment_cost(self, x, y):
        """Calculate the cost (SSE) for a segment using linear regression."""
        n = len(x)
        if n < 2:
            return 0.0
        
        # Calculate means
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        # Calculate coefficients
        x_centered = x - x_mean
        numerator = np.sum(x_centered * (y - y_mean))
        denominator = np.sum(x_centered * x_centered)
        
        # Handle potential numerical instability
        if abs(denominator) < 1e-10:
            return np.sum((y - y_mean) ** 2)
        
        # Calculate slope and intercept
        slope = numerator / denominator
        intercept = y_mean - slope * x_mean
        
        # Calculate residuals and cost
        residuals = y - (slope * x + intercept)
        return np.sum(residuals ** 2)


def analyze_segments(x, y, breakpoints):
    """Analyze the segments to get slopes and other statistics."""
    segments = []
    start_idx = 0
    
    for end_idx in breakpoints + [len(x)]:
        segment_x = x[start_idx:end_idx]
        segment_y = y[start_idx:end_idx]
        
        if len(segment_x) > 1:
            # Fit line to segment
            coeffs = np.polyfit(segment_x, segment_y, 1)
            slope, intercept = coeffs
            
            # Calculate R² and MSE
            y_pred = slope * segment_x + intercept
            residuals = segment_y - y_pred
            mse = np.mean(residuals ** 2)
            r2 = 1 - np.sum(residuals ** 2) / np.sum((segment_y - np.mean(segment_y)) ** 2)
            
            segments.append({
                'start_idx': start_idx,
                'end_idx': end_idx - 1,
                'slope': slope,
                'intercept': intercept,
                'mse': mse,
                'r2': r2,
                'duration': segment_x[-1] - segment_x[0] if len(segment_x) > 0 else 0
            })
        
        start_idx = end_idx
    
    return segments


class CUSUMDetection:
    """
    CUSUM Method for detecting shock waves in traffic data.
    
    This approach uses CUSUM (Cumulative Sum) control charts to detect 
    change points in velocity data without dynamic programming optimization.
    """
    
    def __init__(self, cusum_threshold=7, cusum_drift=1.0, 
                 min_segment_length=10, max_candidates=None,
                 reset_on_detection=True):
        """
        Initialize the CUSUM detector.
        
        Args:
            cusum_threshold: Detection threshold for CUSUM
            cusum_drift: Drift parameter for CUSUM
            min_segment_length: Minimum segment length between change points
            max_candidates: Maximum number of candidate change points
            reset_on_detection: Whether to reset CUSUM statistics after detection
        """
        self.cusum_threshold = cusum_threshold
        self.cusum_drift = cusum_drift
        self.min_segment_length = min_segment_length
        self.max_candidates = max_candidates
        self.reset_on_detection = reset_on_detection
    
    def detect(self, time, data):
        """
        Detect shock waves using the CUSUM method.
        
        Args:
            time: Array of time points
            data: Array of data (velocity, position, etc.)
            
        Returns:
            breakpoints: Indices where shock waves are detected
            diagnostics: Additional information about the detection
        """
        n = len(data)
        if n < 2 * self.min_segment_length:
            return [], {
                "total_cost": 0, 
                "num_segments": 1, 
                "avg_segment_length": n,
                "s_pos": np.zeros(n),
                "s_neg": np.zeros(n)
            }
        
        # Initialize CUSUM statistics
        s_pos = np.zeros(n)
        s_neg = np.zeros(n)
        
        # Compute mean for initial segment
        initial_window = min(self.min_segment_length * 2, n // 4)
        mean_data = np.mean(data[:initial_window])
        
        breakpoints = []
        last_cp = 0
        
        # Compute CUSUM statistics
        for i in range(1, n):
            # Deviation from target mean
            deviation = data[i] - mean_data
            
            # Update positive and negative CUSUMs
            s_pos[i] = max(0, s_pos[i-1] + deviation - self.cusum_drift)
            s_neg[i] = max(0, s_neg[i-1] - deviation - self.cusum_drift)
            
            # Check if either CUSUM exceeds threshold
            if ((s_pos[i] > self.cusum_threshold or s_neg[i] > self.cusum_threshold) 
                and i - last_cp >= self.min_segment_length):
                
                breakpoints.append(i)
                last_cp = i
                
                if self.reset_on_detection:
                    # Reset CUSUM statistics after detection
                    s_pos[i] = 0
                    s_neg[i] = 0
                    
                    # Update mean for next segment
                    next_window_start = min(i + 1, n - self.min_segment_length)
                    next_window_end = min(i + self.min_segment_length + 1, n)
                    
                    if next_window_end > next_window_start:
                        mean_data = np.mean(data[next_window_start:next_window_end])
                    else:
                        mean_data = np.mean(data[i:])
        
        # Apply maximum candidates constraint if specified
        if self.max_candidates is not None and len(breakpoints) > self.max_candidates:
            # Keep the strongest change points based on CUSUM values
            strengths = []
            for bp in breakpoints:
                strength = max(s_pos[bp] if bp < len(s_pos) else 0, 
                             s_neg[bp] if bp < len(s_neg) else 0)
                strengths.append((bp, strength))
            
            # Sort by strength and keep top candidates
            strengths.sort(key=lambda x: x[1], reverse=True)
            breakpoints = sorted([bp for bp, _ in strengths[:self.max_candidates]])
        
        # Calculate total cost as sum of CUSUM statistics
        total_cost = np.sum(s_pos) + np.sum(s_neg)
        
        # Prepare diagnostics
        diagnostics = {
            "s_pos": s_pos,
            "s_neg": s_neg,
            "total_cost": total_cost,
            "num_segments": len(breakpoints) + 1,
            "avg_segment_length": n / (len(breakpoints) + 1),
            "threshold": self.cusum_threshold,
            "drift": self.cusum_drift,
            "mean_baseline": mean_data
        }
        
        print(f"CUSUM detected {len(breakpoints)} breakpoints")
        return breakpoints, diagnostics
    
    def detect_with_data_dict(self, data_dict):
        """
        Alternative interface that accepts a dictionary with time, distance, velocity.
        Uses velocity for CUSUM detection.
        
        Args:
            data_dict: Dictionary containing 'time', 'distance', 'velocity' arrays
            
        Returns:
            breakpoints: Indices where shock waves are detected
            diagnostics: Additional information about the detection
        """
        return self.detect(data_dict['time'], data_dict['velocity'])
    
    def _adaptive_mean_update(self, data, current_idx, window_size):
        """
        Update the baseline mean adaptively based on recent data.
        
        Args:
            data: Full data array
            current_idx: Current position in data
            window_size: Size of window for mean calculation
            
        Returns:
            Updated mean value
        """
        start_idx = max(0, current_idx - window_size)
        end_idx = min(len(data), current_idx + 1)
        return np.mean(data[start_idx:end_idx])


class AdaptiveCUSUMDetection(CUSUMDetection):
    """
    Adaptive CUSUM Detection that updates the baseline mean dynamically.
    
    This variant of CUSUM detection continuously updates the baseline mean
    based on recent observations, making it more sensitive to gradual changes.
    """
    
    def __init__(self, cusum_threshold=5, cusum_drift=0.5, 
                 min_segment_length=10, max_candidates=None,
                 adaptation_window=20, adaptation_rate=0.1):
        """
        Initialize the Adaptive CUSUM detector.
        
        Args:
            cusum_threshold: Detection threshold for CUSUM (lower for more sensitivity)
            cusum_drift: Drift parameter for CUSUM
            min_segment_length: Minimum segment length between change points
            max_candidates: Maximum number of candidate change points
            adaptation_window: Window size for adaptive mean update
            adaptation_rate: Rate of adaptation (0-1, higher = more adaptive)
        """
        super().__init__(cusum_threshold, cusum_drift, min_segment_length, 
                        max_candidates, reset_on_detection=False)
        self.adaptation_window = adaptation_window
        self.adaptation_rate = adaptation_rate
    
    def detect(self, time, data):
        """
        Detect shock waves using adaptive CUSUM method.
        """
        n = len(data)
        if n < 2 * self.min_segment_length:
            return [], {
                "total_cost": 0, 
                "num_segments": 1, 
                "avg_segment_length": n,
                "s_pos": np.zeros(n),
                "s_neg": np.zeros(n)
            }
        
        # Initialize
        s_pos = np.zeros(n)
        s_neg = np.zeros(n)
        adaptive_mean = np.mean(data[:self.adaptation_window])
        
        breakpoints = []
        last_cp = 0
        
        # Track adaptive mean over time
        mean_history = np.zeros(n)
        mean_history[0] = adaptive_mean
        
        for i in range(1, n):
            # Update adaptive mean
            if i >= self.adaptation_window:
                window_mean = np.mean(data[i-self.adaptation_window:i])
                adaptive_mean = ((1 - self.adaptation_rate) * adaptive_mean + 
                               self.adaptation_rate * window_mean)
            
            mean_history[i] = adaptive_mean
            
            # Deviation from adaptive mean
            deviation = data[i] - adaptive_mean
            
            # Update CUSUM statistics
            s_pos[i] = max(0, s_pos[i-1] + deviation - self.cusum_drift)
            s_neg[i] = max(0, s_neg[i-1] - deviation - self.cusum_drift)
            
            # Check for change point
            if ((s_pos[i] > self.cusum_threshold or s_neg[i] > self.cusum_threshold) 
                and i - last_cp >= self.min_segment_length):
                
                breakpoints.append(i)
                last_cp = i
                
                # Partial reset - reduce but don't zero out CUSUM values
                s_pos[i] *= 0.3
                s_neg[i] *= 0.3
        
        # Apply maximum candidates constraint
        if self.max_candidates is not None and len(breakpoints) > self.max_candidates:
            strengths = [(bp, max(s_pos[bp], s_neg[bp])) for bp in breakpoints]
            strengths.sort(key=lambda x: x[1], reverse=True)
            breakpoints = sorted([bp for bp, _ in strengths[:self.max_candidates]])
        
        # Enhanced diagnostics
        diagnostics = {
            "s_pos": s_pos,
            "s_neg": s_neg,
            "adaptive_mean": mean_history,
            "total_cost": np.sum(s_pos) + np.sum(s_neg),
            "num_segments": len(breakpoints) + 1,
            "avg_segment_length": n / (len(breakpoints) + 1),
            "threshold": self.cusum_threshold,
            "drift": self.cusum_drift,
            "adaptation_window": self.adaptation_window,
            "adaptation_rate": self.adaptation_rate
        }
        
        print(f"Adaptive CUSUM detected {len(breakpoints)} breakpoints")
        return breakpoints, diagnostics