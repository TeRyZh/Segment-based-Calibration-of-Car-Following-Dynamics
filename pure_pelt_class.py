import numpy as np
from typing import List, Tuple, Dict, Any


class PurePELTDetection:
    """
    Pure PELT (Pruned Exact Linear Time) Method for detecting change points in traffic data.
    
    This implementation uses the standard PELT algorithm without CUSUM candidate generation,
    considering all possible change points. Has O(n²) worst-case complexity but uses pruning
    to achieve near-linear performance in practice.
    """
    
    def __init__(self, penalty=50, min_segment_length=10, max_changepoints=None,
                 cost_function='normal_var'):
        """
        Initialize the PELT detector.
        
        Args:
            penalty: Penalty parameter for adding change points (larger = fewer change points)
            min_segment_length: Minimum segment length between change points
            max_changepoints: Maximum number of change points to detect
            cost_function: Cost function to use ('normal_var', 'normal_mean', 'poisson')
        """
        self.penalty = penalty
        self.min_segment_length = min_segment_length
        self.max_changepoints = max_changepoints
        self.cost_function = cost_function
    
    def detect(self, data):
        """
        Detect change points using the pure PELT method.
        
        Args:
            data: Dictionary containing 'time', 'distance', 'velocity' arrays
            
        Returns:
            optimal_changepoints: Indices where change points are detected
            diagnostics: Additional information about the detection
        """
        # Apply PELT algorithm to velocity data
        optimal_changepoints, pelt_costs = self._pelt_algorithm(
            data['distance'],
            penalty=self.penalty,
            min_segment_length=self.min_segment_length,
            max_changepoints=self.max_changepoints
        )
        
        # Calculate slope and other diagnostics for each detected segment
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
            "pelt": {
                "changepoints": optimal_changepoints,
                "segment_costs": pelt_costs,
                "penalty_used": self.penalty,
                "cost_function": self.cost_function
            },
            "complexity": {
                "algorithm": "pure_pelt",
                "data_points": len(data['velocity']),
                "theoretical_operations": len(data['velocity']) ** 2,
                "pruning_efficiency": pelt_costs.get("pruning_efficiency", "N/A")
            },
            "num_segments": len(optimal_changepoints) + 1,
            "avg_segment_length": len(data['time']) / (len(optimal_changepoints) + 1) if optimal_changepoints else len(data['time']),
            "segments": segments
        }
        
        return optimal_changepoints, diagnostics
    
    def _pelt_algorithm(self, data, penalty, min_segment_length, max_changepoints=None):
        """
        Implementation of the pure PELT algorithm.
        
        Args:
            data: 1D array of observations (velocity data)
            penalty: Penalty parameter
            min_segment_length: Minimum segment length
            max_changepoints: Maximum number of change points
            
        Returns:
            changepoints: List of detected change point indices
            costs: Dictionary of costs for diagnostics
        """
        n = len(data)
        if n < 2 * min_segment_length:
            return [], {"total_cost": 0, "pruning_efficiency": 1.0}
        
        # Initialize dynamic programming arrays
        F = np.full(n + 1, np.inf)  # Optimal cost up to point i
        F[0] = -penalty  # Base case (starting point)
        R = set([0])  # Pruning set (active set of potential change points)
        cp_track = {}  # Track change points for reconstruction
        
        # Track pruning efficiency
        total_evaluations = 0
        pruned_evaluations = 0
        
        # PELT main loop
        for t in range(min_segment_length, n + 1):
            # Find optimal previous change point
            candidates = []
            R_to_remove = set()
            
            for s in R:
                if t - s >= min_segment_length:
                    total_evaluations += 1
                    
                    # Calculate cost for segment [s, t)
                    if s < t:
                        segment = data[s:t]
                        segment_cost = self._calculate_cost(segment)
                        cost = F[s] + segment_cost + penalty
                        candidates.append((cost, s))
                    
                    # Pruning: remove s if it can never be optimal for future points
                    # This is the key efficiency gain of PELT
                    if F[s] + self._calculate_cost(data[s:t]) > F[t-1]:
                        R_to_remove.add(s)
                        pruned_evaluations += 1
            
            # Remove pruned points
            R -= R_to_remove
            
            if candidates:
                F[t], best_s = min(candidates)
                cp_track[t] = best_s
            
            # Add current point to active set
            R.add(t)
        
        # Calculate pruning efficiency
        pruning_efficiency = pruned_evaluations / max(total_evaluations, 1)
        
        # Reconstruct change points
        changepoints = self._reconstruct_changepoints(cp_track, n)
        
        # Apply maximum change points constraint if specified
        if max_changepoints is not None and len(changepoints) > max_changepoints:
            changepoints = self._select_best_changepoints(
                data, changepoints, max_changepoints
            )
        
        # Prepare cost diagnostics
        costs = {
            "total_cost": F[n],
            "total_evaluations": total_evaluations,
            "pruned_evaluations": pruned_evaluations,
            "pruning_efficiency": pruning_efficiency,
            "final_active_set_size": len(R)
        }
        
        return changepoints, costs
    
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
            var = np.var(segment)
            if var <= 0:
                return 0.0
            return len(segment) * np.log(var)
        
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
            var = np.var(segment)
            if var <= 0:
                return 0.0
            return len(segment) * np.log(var)
    
    def _reconstruct_changepoints(self, cp_track, n):
        """
        Reconstruct the optimal sequence of change points.
        
        Args:
            cp_track: Dictionary tracking optimal previous change point indices
            n: Length of data
            
        Returns:
            List of change point indices
        """
        changepoints = []
        current = n
        
        while current in cp_track and cp_track[current] != 0:
            prev_idx = cp_track[current]
            if prev_idx > 0:  # Don't include the starting point (0)
                changepoints.append(prev_idx)
            current = prev_idx
        
        return sorted(changepoints)
    
    def _select_best_changepoints(self, data, changepoints, max_changepoints):
        """
        Select the best subset of change points if there are too many.
        
        Args:
            data: Original data array
            changepoints: List of all detected change points
            max_changepoints: Maximum number to keep
            
        Returns:
            List of best change points
        """
        if len(changepoints) <= max_changepoints:
            return changepoints
        
        # Calculate the "strength" of each change point based on cost reduction
        strengths = []
        all_points = [0] + sorted(changepoints) + [len(data)]
        
        for i, cp in enumerate(changepoints):
            # Find position in the sorted list
            cp_pos = sorted(changepoints).index(cp)
            
            # Get adjacent points
            start_point = all_points[cp_pos]
            end_point = all_points[cp_pos + 2]
            
            # Cost with change point (two segments)
            cost_with = (self._calculate_cost(data[start_point:cp]) + 
                        self._calculate_cost(data[cp:end_point]))
            
            # Cost without change point (one segment)
            cost_without = self._calculate_cost(data[start_point:end_point])
            
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
            'cost_function': self.cost_function
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
    
    # Initialize Pure PELT detector
    detector = PurePELTDetection(
        penalty=75,
        min_segment_length=20,
        cost_function='normal_var'
    )
    
    # Detect change points
    changepoints, diagnostics = detector.detect(test_data)
    
    print("Pure PELT Detection Results:")
    print(f"Detected change points: {changepoints}")
    print(f"True change points: [300, 600, 800]")
    print(f"Number of segments: {diagnostics['num_segments']}")
    print(f"Average segment length: {diagnostics['avg_segment_length']:.1f}")
    print(f"Pruning efficiency: {diagnostics['complexity']['pruning_efficiency']:.2%}")
    print(f"Total evaluations: {diagnostics['pelt']['segment_costs'].get('total_evaluations', 'N/A')}")
