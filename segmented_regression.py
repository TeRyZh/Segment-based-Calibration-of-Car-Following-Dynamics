import numpy as np
from numba import jit
import warnings



def calculate_segment_cost(x, y):
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

def segmented_regression_dp(x, y, penalty, min_segment_length=20):
    """Perform segmented regression using dynamic programming."""
    n = len(x)
    if n < 2 * min_segment_length:
        return [], {"total_cost": 0, "num_segments": 1, "avg_segment_length": n}
    
    # Initialize DP arrays
    dp = np.full(n, np.inf)  # Minimum cost up to point i
    prev = np.full(n, -1, dtype=int)  # Previous breakpoint for reconstruction
    
    # Precompute segment costs
    segment_costs = np.full((n, n), np.inf)
    for i in range(n):
        for j in range(i + min_segment_length - 1, n):
            segment_costs[i, j] = calculate_segment_cost(x[i:j+1], y[i:j+1])
    
    # Base cases
    for i in range(min_segment_length - 1, n):
        dp[i] = segment_costs[0, i]
    
    # Dynamic Programming
    for i in range(min_segment_length, n):
        for j in range(min_segment_length - 1, i - min_segment_length + 1):
            cost = dp[j] + penalty + segment_costs[j+1, i]
            if cost < dp[i]:
                dp[i] = cost
                prev[i] = j
    
    # Reconstruct solution
    breakpoints = []
    curr = n - 1
    while curr > 0:
        if prev[curr] != -1:
            breakpoints.append(prev[curr] + 1)
            curr = prev[curr]
        else:
            break
    
    breakpoints = sorted(breakpoints)
    
    # Compute diagnostics
    diagnostics = {
        "total_cost": dp[n-1],
        "num_segments": len(breakpoints) + 1,
        "avg_segment_length": n / (len(breakpoints) + 1)
    }
    
    return breakpoints, diagnostics


def generate_cusum_candidates(velocity, cusum_threshold=7, cusum_drift=1.0, 
                                          min_segment_length=20, max_candidates=None):
    """
    Generate candidate change points using CUSUM on velocity data.
    
    Args:
        x: Time points
        y: Position values
        cusum_threshold: Detection threshold for CUSUM
        cusum_drift: Drift parameter for CUSUM
        min_segment_length: Minimum segment length
        max_candidates: Maximum number of candidate change points to consider
        
    Returns:
        List of candidate changepoint indices
    """
    n = len(velocity)
    if n < 2 * min_segment_length:
        return []
    
    # Now run CUSUM on velocity data
    S_pos = np.zeros(n)
    S_neg = np.zeros(n)
    
    # Compute mean for initial segment
    mean_velocity = np.mean(velocity[:min_segment_length])
    
    last_cp = 0
    candidates = []
    
    # Compute CUSUM statistics on velocity
    for i in range(min_segment_length, n):
        # Deviation from target velocity
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
            
            # Update mean velocity for next segment
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


def select_optimal_changepoints(x, y, candidates, dp_penalty=100, min_segment_length=10):
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
                segment_costs[(i, j)] = calculate_segment_cost(x[start:end+1], y[start:end+1])
    
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


def cusum_dp(x, y, v, cusum_threshold=7, cusum_drift=1.0, 
             dp_penalty=100, min_cusum_length=10, min_segment_length=10,
             max_candidates=None):
    """
    Hybrid change point detection combining CUSUM for candidate generation
    with dynamic programming for optimal selection.
    
    Args:
        x: Time points
        y: Position values
        cusum_threshold: Detection threshold for CUSUM
        cusum_drift: Drift parameter for CUSUM
        dp_penalty: Penalty value for adding a changepoint in DP
        min_segment_length: Minimum segment length
        max_candidates: Maximum number of candidate change points to consider
        
    Returns:
        List of optimal changepoint indices
    """
    # Step 1: Generate candidate change points using CUSUM
    candidates, s_pos, s_neg = generate_cusum_candidates(
        v, 
        cusum_threshold=cusum_threshold,
        cusum_drift=cusum_drift,
        min_segment_length=min_cusum_length,
        max_candidates=max_candidates
    )
    
    # Step 2: Use dynamic programming to find optimal subset of candidates
    optimal_changepoints = select_optimal_changepoints(
        x, y, candidates,
        dp_penalty=dp_penalty,
        min_segment_length=min_segment_length
    )


    # Prepare diagnostics
    diagnostics = {
        "cusum": {
            "s_pos": s_pos,
            "s_neg": s_neg,
            "candidates": candidates
        },
        "num_segments": len(optimal_changepoints) + 1,
        "avg_segment_length": len(x) / (len(optimal_changepoints) + 1)
    }

    
    return optimal_changepoints, diagnostics


def analyze_segments(x, y, breakpoints):
    """Analyze the segments to get slopes and other statistics."""
    segments = []
    start_idx = 0
    
    for end_idx in breakpoints + [len(x)]:
        segment_x = x[start_idx:end_idx]
        segment_y = y[start_idx:end_idx]
        
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
            'duration': segment_x[-1] - segment_x[0]
        })
        
        start_idx = end_idx
    
    return segments

def analyze_vehicle_trajectory(vehicle_data, penalty=150, min_segment_length=20):
    """Analyze a single vehicle's trajectory."""
    time = vehicle_data['Frame_ID'].values / 10  # Convert to seconds
    position = vehicle_data['Local_Y'].values 
    
    # Perform segmented regression
    breakpoints, diagnostics = segmented_regression_dp(
        time,
        position,
        penalty=penalty,
        min_segment_length=min_segment_length
    )
    
    # Analyze segments
    segments = analyze_segments(time, position, breakpoints) 
    
    return time, position, breakpoints, segments, diagnostics 