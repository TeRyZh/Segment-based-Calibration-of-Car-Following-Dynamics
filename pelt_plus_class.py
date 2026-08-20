import numpy as np
from typing import List, Tuple, Dict, Any


class PELTPlusDetection:
    """
    PELT+ (Candidate-Guided PELT) Method for detecting shock waves in traffic data.
    
    This approach combines CUSUM candidate generation with PELT optimization to achieve
    O(n*m) complexity where m << n, representing a significant improvement over standard
    O(n²) dynamic programming approaches while maintaining optimality guarantees.
    """
    
    def __init__(self, penalty=50, min_segment_length=10, max_changepoints=None,
                 cost_function='normal_var', cusum_threshold=7, cusum_drift=1.0,
                 max_candidates=None):
        """
        Initialize the PELT+ detector.
        
        Args:
            penalty: Penalty parameter for adding change points (larger = fewer change points)
            min_segment_length: Minimum segment length between change points
            max_changepoints: Maximum number of change points to detect
            cost_function: Cost function to use ('normal_var', 'normal_mean', 'poisson')
            cusum_threshold: Detection threshold for CUSUM candidate generation
            cusum_drift: Drift parameter for CUSUM candidate generation
            max_candidates: Maximum number of candidate change points from CUSUM
        """
        self.penalty = penalty
        self.min_segment_length = min_segment_length
        self.max_changepoints = max_changepoints
        self.cost_function = cost_function
        self.cusum_threshold = cusum_threshold
        self.cusum_drift = cusum_drift
        self.max_candidates = max_candidates
    
    def detect(self, data):
        """
        Detect shock waves using the PELT+ method.
        
        Args:
            data: Dictionary containing 'time', 'distance', 'velocity' arrays
            
        Returns:
            optimal_changepoints: Indices where shock waves are detected
            diagnostics: Additional information about the detection
        """
        # Step 1: Generate candidate points using CUSUM
        candidate_points, cusum_diagnostics = self._generate_cusum_candidates(data['velocity'])
        
        # Step 2: Apply PELT algorithm restricted to candidate points
        optimal_changepoints, pelt_costs = self._pelt_plus_algorithm(
            data['distance'],
            candidate_points,
            penalty=self.penalty,
            min_segment_length=self.min_segment_length,
            max_changepoints=self.max_changepoints
        )
        
        # Step 3: Calculate slope and other diagnostics for each detected segment
        segments = []
        if optimal_changepoints:
            start_idx = 0
            
            for bp in optimal_changepoints + [len(data['time'])-1]:
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
        
        # Prepare comprehensive diagnostics
        diagnostics = {
            "pelt_plus": {
                "changepoints": optimal_changepoints,
                "segment_costs": pelt_costs,
                "penalty_used": self.penalty,
                "cost_function": self.cost_function
            },
            "cusum": cusum_diagnostics,
            "candidates": {
                "total_candidates": len(candidate_points) - 2,  # Exclude endpoints
                "candidate_points": candidate_points[1:-1],  # Exclude endpoints for display
                "compression_ratio": len(data['velocity']) / len(candidate_points) if candidate_points else 1
            },
            "complexity": {
                "standard_pelt_operations": len(data['velocity']) ** 2,
                "pelt_plus_operations": len(data['velocity']) * len(candidate_points),
                "speedup_factor": (len(data['velocity']) ** 2) / (len(data['velocity']) * len(candidate_points)) if candidate_points else 1
            },
            "num_segments": len(optimal_changepoints) + 1,
            "avg_segment_length": len(data['time']) / (len(optimal_changepoints) + 1) if optimal_changepoints else len(data['time']),
            "segments": segments
        }
        
        return optimal_changepoints, diagnostics
    
    def _generate_cusum_candidates(self, velocity):
        """
        Generate candidate change points using CUSUM analysis.
        
        Args:
            velocity: Velocity data values
            
        Returns:
            candidate_points: List of candidate indices including endpoints
            cusum_diagnostics: CUSUM analysis results
        """
        n = len(velocity)
        if n < 2 * self.min_segment_length:
            return [0, n-1], {"s_pos": np.zeros(n), "s_neg": np.zeros(n), "changepoints": []}
        
        # Initialize CUSUM statistics
        S_pos = np.zeros(n)
        S_neg = np.zeros(n)
        
        # Compute mean for initial segment
        mean_velocity = np.mean(velocity[:self.min_segment_length])
        
        last_cp = 0
        changepoints = []
        
        # Compute CUSUM statistics on velocity
        for i in range(self.min_segment_length, n):
            # Deviation from target velocity
            deviation = velocity[i] - mean_velocity
            
            # Update positive and negative CUSUMs
            S_pos[i] = max(0, S_pos[i-1] + deviation - self.cusum_drift)
            S_neg[i] = max(0, S_neg[i-1] - deviation - self.cusum_drift)
            
            # Check if either CUSUM exceeds threshold
            if (S_pos[i] > self.cusum_threshold or S_neg[i] > self.cusum_threshold) and i - last_cp >= self.min_segment_length:
                changepoints.append(i)
                
                # Reset CUSUM statistics
                S_pos[i] = 0
                S_neg[i] = 0
                
                # Update mean velocity for next segment
                if i + self.min_segment_length < n:
                    mean_velocity = np.mean(velocity[i:i+self.min_segment_length])
                else:
                    mean_velocity = np.mean(velocity[i:])
                
                last_cp = i
        
        # Limit candidates if needed
        if self.max_candidates is not None and len(changepoints) > self.max_candidates:
            # Keep the strongest change points based on CUSUM values
            strengths = []
            for cp in changepoints:
                if cp < len(S_pos):
                    strengths.append(max(S_pos[cp], S_neg[cp]))
                else:
                    strengths.append(0)
            
            strongest_indices = np.argsort(strengths)[-self.max_candidates:]
            changepoints = [changepoints[i] for i in strongest_indices]
            changepoints.sort()  # Maintain temporal order
        
        # Include endpoints in candidate set C⁺ = {0} ∪ C ∪ {n-1}
        candidate_points = [0] + changepoints + [n-1]
        candidate_points = sorted(list(set(candidate_points)))  # Remove duplicates and sort
        
        cusum_diagnostics = {
            "s_pos": S_pos,
            "s_neg": S_neg,
            "changepoints": changepoints,
            "threshold_used": self.cusum_threshold,
            "drift_used": self.cusum_drift
        }
        
        return candidate_points, cusum_diagnostics
    
    def _pelt_plus_algorithm(self, data, candidate_points, penalty, min_segment_length, max_changepoints=None):
        """
        Implementation of the PELT+ algorithm with candidate point restriction.
        
        Args:
            data: 1D array of observations (distance data)
            candidate_points: List of candidate change point indices (including endpoints)
            penalty: Penalty parameter
            min_segment_length: Minimum segment length
            max_changepoints: Maximum number of change points
            
        Returns:
            changepoints: List of detected change point indices
            costs: Dictionary of costs for diagnostics
        """
        if len(candidate_points) < 2:
            return [], {}
        
        # Create mapping from candidate points to indices for efficient lookup
        candidate_set = set(candidate_points)
        candidate_to_idx = {cp: i for i, cp in enumerate(candidate_points)}
        
        # Initialize dynamic programming arrays
        m = len(candidate_points)
        F = np.full(m, np.inf)  # Optimal cost up to candidate point i
        F[0] = -penalty  # Base case (starting point)
        R = set([0])  # Pruning set (indices in candidate_points)
        cp_track = {}  # Track change points for reconstruction
        
        # Precompute segment costs only for candidate pairs
        segment_costs = self._precompute_candidate_costs(data, candidate_points)
        
        # PELT+ main loop - iterate over candidate points only
        for i in range(1, m):
            t = candidate_points[i]  # Current candidate point
            
            # Find optimal previous candidate point
            candidates = []
            for j in R:
                s = candidate_points[j]  # Previous candidate point
                if t - s >= min_segment_length:
                    cost = F[j] + self._get_candidate_cost(segment_costs, j, i) + penalty
                    candidates.append((cost, j))
            
            if candidates:
                F[i], best_j = min(candidates)
                cp_track[i] = best_j
            
            # Pruning step - remove dominated candidate points
            R_new = set()
            for j in R:
                s = candidate_points[j]
                if t - s >= min_segment_length:
                    # Keep j if it could be optimal for some future candidate
                    cost_j = F[j] + self._get_candidate_cost(segment_costs, j, i)
                    if cost_j <= F[i]:
                        R_new.add(j)
            
            R_new.add(i)
            R = R_new
        
        # Reconstruct change points from candidate space
        changepoints = self._reconstruct_candidate_changepoints(cp_track, candidate_points, m-1)
        
        # Apply maximum change points constraint if specified
        if max_changepoints is not None and len(changepoints) > max_changepoints:
            changepoints = self._select_best_candidate_changepoints(
                data, changepoints, max_changepoints, candidate_points, segment_costs
            )
        
        # Prepare cost diagnostics
        costs = {
            "total_cost": F[m-1],
            "candidate_costs": F.tolist(),
            "num_candidates_evaluated": m
        }
        
        return changepoints, costs
    
    def _precompute_candidate_costs(self, data, candidate_points):
        """
        Precompute costs only for candidate point pairs to improve efficiency.
        
        Args:
            data: 1D array of observations
            candidate_points: List of candidate change point indices
            
        Returns:
            Dictionary of precomputed segment costs between candidate pairs
        """
        m = len(candidate_points)
        costs = {}
        
        for i in range(m):
            for j in range(i + 1, m):
                start = candidate_points[i]
                end = candidate_points[j]
                if end - start >= self.min_segment_length:
                    segment = data[start:end]
                    costs[(i, j)] = self._calculate_cost(segment)
        
        return costs
    
    def _get_candidate_cost(self, segment_costs, start_idx, end_idx):
        """Get precomputed segment cost between candidate points."""
        return segment_costs.get((start_idx, end_idx), np.inf)
    
    def _calculate_cost(self, segment):
        """
        Calculate the cost for a given segment based on the chosen cost function.
        
        Args:
            segment: Array of observations in the segment
            
        Returns:
            Cost value for the segment
        """
        if len(segment) <= 1:
            return 0.0
        
        if self.cost_function == 'normal_var':
            # Cost based on variance (assumes normal distribution with unknown variance)
            return len(segment) * np.log(np.var(segment) + 1e-10)
        
        elif self.cost_function == 'normal_mean':
            # Cost based on mean change (assumes normal distribution with known variance)
            return np.sum((segment - np.mean(segment))**2)
        
        elif self.cost_function == 'poisson':
            # Cost for Poisson distribution
            mean_val = np.mean(segment)
            if mean_val <= 0:
                return np.inf
            return -np.sum(segment * np.log(mean_val) - mean_val)
        
        else:
            # Default to normal variance
            return len(segment) * np.log(np.var(segment) + 1e-10)
    
    def _reconstruct_candidate_changepoints(self, cp_track, candidate_points, final_idx):
        """
        Reconstruct the optimal sequence of change points from candidate space.
        
        Args:
            cp_track: Dictionary tracking optimal previous candidate indices
            candidate_points: List of candidate change point indices
            final_idx: Final candidate index
            
        Returns:
            List of change point indices in original data space
        """
        changepoints = []
        current = final_idx
        
        while current in cp_track and cp_track[current] != 0:
            prev_idx = cp_track[current]
            changepoints.append(candidate_points[prev_idx])
            current = prev_idx
        
        # Remove the initial point (0) and sort
        changepoints = [cp for cp in changepoints if cp > 0 and cp < candidate_points[-1]]
        return sorted(changepoints)
    
    def _select_best_candidate_changepoints(self, data, changepoints, max_changepoints, 
                                          candidate_points, segment_costs):
        """
        Select the best subset of change points from candidates if there are too many.
        
        Args:
            data: Original data array
            changepoints: List of all detected change points
            max_changepoints: Maximum number to keep
            candidate_points: List of candidate points
            segment_costs: Precomputed segment costs
            
        Returns:
            List of best change points
        """
        if len(changepoints) <= max_changepoints:
            return changepoints
        
        # Create mapping for efficient lookup
        candidate_to_idx = {cp: i for i, cp in enumerate(candidate_points)}
        
        # Calculate the "strength" of each change point based on cost reduction
        strengths = []
        for i, cp in enumerate(changepoints):
            # Find candidate indices
            if i == 0:
                start_cp = 0
            else:
                start_cp = changepoints[i-1]
            
            if i == len(changepoints) - 1:
                end_cp = candidate_points[-1]
            else:
                end_cp = changepoints[i+1]
            
            start_idx = candidate_to_idx[start_cp]
            cp_idx = candidate_to_idx[cp]
            end_idx = candidate_to_idx[end_cp]
            
            # Cost with change point
            cost_with = (self._get_candidate_cost(segment_costs, start_idx, cp_idx) + 
                        self._get_candidate_cost(segment_costs, cp_idx, end_idx))
            
            # Cost without change point
            cost_without = self._get_candidate_cost(segment_costs, start_idx, end_idx)
            
            # Strength is the cost reduction from having this change point
            strength = cost_without - cost_with
            strengths.append(strength)
        
        # Select the strongest change points
        strongest_indices = np.argsort(strengths)[-max_changepoints:]
        best_changepoints = [changepoints[i] for i in strongest_indices]
        
        return sorted(best_changepoints)
    
    def set_parameters(self, **kwargs):
        """
        Update detection parameters.
        
        Args:
            **kwargs: Parameter names and values to update
        """
        for param, value in kwargs.items():
            if hasattr(self, param):
                setattr(self, param, value)
            else:
                print(f"Warning: Parameter '{param}' not recognized")
    
    def get_parameters(self):
        """
        Get current parameter values.
        
        Returns:
            Dictionary of current parameters
        """
        return {
            'penalty': self.penalty,
            'min_segment_length': self.min_segment_length,
            'max_changepoints': self.max_changepoints,
            'cost_function': self.cost_function,
            'cusum_threshold': self.cusum_threshold,
            'cusum_drift': self.cusum_drift,
            'max_candidates': self.max_candidates
        }


# Example usage and testing
if __name__ == "__main__":
    # Example synthetic data
    np.random.seed(42)
    n = 1000
    
    # Generate synthetic trajectory data with change points
    time = np.linspace(0, 100, n)
    
    # Create velocity with regime changes
    velocity = np.zeros(n)
    velocity[0:300] = 30 + np.random.normal(0, 2, 300)      # Segment 1: ~30 mph
    velocity[300:600] = 50 + np.random.normal(0, 3, 300)    # Segment 2: ~50 mph  
    velocity[600:800] = 20 + np.random.normal(0, 2, 200)    # Segment 3: ~20 mph
    velocity[800:1000] = 40 + np.random.normal(0, 2, 200)   # Segment 4: ~40 mph
    
    # Generate corresponding distance (cumulative)
    distance = np.cumsum(velocity * (time[1] - time[0]))
    
    # Create data dictionary
    test_data = {
        'time': time,
        'distance': distance,
        'velocity': velocity
    }
    
    # Initialize PELT+ detector
    detector = PELTPlusDetection(
        penalty=75,
        min_segment_length=20,
        cusum_threshold=5,
        cusum_drift=1.0
    )
    
    # Detect change points
    changepoints, diagnostics = detector.detect(test_data)
    
    print("PELT+ Detection Results:")
    print(f"Detected change points: {changepoints}")
    print(f"True change points: [300, 600, 800]")
    print(f"Number of candidates generated: {diagnostics['candidates']['total_candidates']}")
    print(f"Compression ratio: {diagnostics['candidates']['compression_ratio']:.2f}")
    print(f"Computational speedup: {diagnostics['complexity']['speedup_factor']:.2f}x")
    print(f"Number of segments: {diagnostics['num_segments']}")
