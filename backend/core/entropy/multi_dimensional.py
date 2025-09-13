"""
Multi-Dimensional Entropy Analysis for CEGO optimization.

This module extends entropy calculation beyond simple content similarity to include:
- Semantic entropy: Content meaning and topic distribution
- Temporal entropy: Time-based patterns and recency
- Relational entropy: Document relationships and dependencies
- Structural entropy: Format and organization patterns

Mathematical Foundation:
H_total = Σᵢ wᵢ × Hᵢ where wᵢ are learned weights for dimension i
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timezone
from collections import Counter, defaultdict
from dataclasses import dataclass
import logging

from .shannon_calculator import ShannonEntropyCalculator, EntropyResult

logger = logging.getLogger(__name__)


@dataclass
class MultiDimensionalResult:
    """Result of multi-dimensional entropy analysis."""
    total_entropy: float
    dimension_entropies: Dict[str, float]
    dimension_weights: Dict[str, float]
    confidence: float
    processing_time_ms: float
    metadata: Dict[str, Any]


@dataclass
class ContentPiece:
    """Extended content representation with metadata for multi-dimensional analysis."""
    text: str
    timestamp: Optional[datetime] = None
    source: Optional[str] = None
    content_type: Optional[str] = None
    relationships: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class MultiDimensionalEntropy:
    """
    Multi-dimensional entropy calculator for comprehensive content analysis.
    
    This calculator considers multiple dimensions of content organization
    to provide more nuanced entropy measurements than single-dimension approaches.
    """
    
    def __init__(self):
        self.shannon_calculator = ShannonEntropyCalculator()
        
        # Default dimension weights (can be learned/adapted)
        self.dimension_weights = {
            'semantic': 0.4,      # Content meaning and similarity
            'temporal': 0.2,      # Time-based patterns
            'relational': 0.2,    # Document relationships
            'structural': 0.2     # Format and organization
        }
        
        # Caching for performance
        self._cache = {}
        
    def calculate_multidimensional_entropy(self, 
                                         content_pieces: List[ContentPiece],
                                         custom_weights: Optional[Dict[str, float]] = None) -> MultiDimensionalResult:
        """
        Calculate entropy across multiple dimensions.
        
        Args:
            content_pieces: List of content with metadata
            custom_weights: Optional custom dimension weights
            
        Returns:
            MultiDimensionalResult with entropy breakdown
            
        Example:
            >>> calculator = MultiDimensionalEntropy()
            >>> pieces = [ContentPiece("ML is great", datetime.now(), "docs")]
            >>> result = calculator.calculate_multidimensional_entropy(pieces)
            >>> print(f"Total entropy: {result.total_entropy:.3f}")
        """
        import time
        start_time = time.time()
        
        if not content_pieces:
            return MultiDimensionalResult(0.0, {}, {}, 0.0, 0.0, {})
        
        # Use custom weights if provided
        weights = custom_weights or self.dimension_weights
        
        # Calculate entropy for each dimension
        dimension_entropies = {}
        
        # Semantic entropy (content similarity)
        dimension_entropies['semantic'] = self._calculate_semantic_entropy(content_pieces)
        
        # Temporal entropy (time-based patterns)
        dimension_entropies['temporal'] = self._calculate_temporal_entropy(content_pieces)
        
        # Relational entropy (document relationships)
        dimension_entropies['relational'] = self._calculate_relational_entropy(content_pieces)
        
        # Structural entropy (format patterns)
        dimension_entropies['structural'] = self._calculate_structural_entropy(content_pieces)
        
        # Calculate weighted total entropy
        total_entropy = sum(
            weights.get(dim, 0) * entropy 
            for dim, entropy in dimension_entropies.items()
        )
        
        # Calculate confidence based on data availability
        confidence = self._calculate_multidimensional_confidence(content_pieces, dimension_entropies)
        
        # Gather metadata
        metadata = {
            'pieces_analyzed': len(content_pieces),
            'dimensions_used': list(dimension_entropies.keys()),
            'weight_scheme': 'custom' if custom_weights else 'default',
            'dominant_dimension': max(dimension_entropies.items(), key=lambda x: x[1])[0]
        }
        
        processing_time = (time.time() - start_time) * 1000
        
        return MultiDimensionalResult(
            total_entropy=total_entropy,
            dimension_entropies=dimension_entropies,
            dimension_weights=weights,
            confidence=confidence,
            processing_time_ms=processing_time,
            metadata=metadata
        )
    
    def _calculate_semantic_entropy(self, content_pieces: List[ContentPiece]) -> float:
        """
        Calculate semantic entropy based on content meaning and topic distribution.
        
        Uses bag-of-words similarity to measure semantic diversity.
        """
        if len(content_pieces) <= 1:
            return 0.0
        
        texts = [piece.text for piece in content_pieces]
        
        # Use Shannon calculator with simple embeddings
        try:
            result = self.shannon_calculator.calculate_content_entropy(texts)
            return result.value
        except Exception as e:
            logger.warning(f"Semantic entropy calculation failed: {e}")
            return 0.0
    
    def _calculate_temporal_entropy(self, content_pieces: List[ContentPiece]) -> float:
        """
        Calculate temporal entropy based on time distribution patterns.
        
        Measures how content is distributed across time periods.
        """
        timestamps = [piece.timestamp for piece in content_pieces if piece.timestamp]
        
        if len(timestamps) <= 1:
            return 0.0
        
        try:
            # Convert to hours since earliest timestamp for analysis
            earliest = min(timestamps)
            hours_since_start = [
                (ts - earliest).total_seconds() / 3600 
                for ts in timestamps
            ]
            
            # Create time bins (e.g., hourly, daily)
            max_hours = max(hours_since_start) if hours_since_start else 1
            num_bins = min(24, int(max_hours) + 1)  # Max 24 bins
            
            if num_bins <= 1:
                return 0.0
            
            # Count documents in each time bin
            bin_counts = np.zeros(num_bins)
            bin_size = max_hours / num_bins
            
            for hours in hours_since_start:
                bin_idx = min(int(hours / bin_size), num_bins - 1)
                bin_counts[bin_idx] += 1
            
            # Convert to probabilities and calculate entropy
            probabilities = bin_counts / np.sum(bin_counts)
            probabilities = probabilities[probabilities > 0]  # Remove zero bins
            
            if len(probabilities) <= 1:
                return 0.0
            
            entropy = -np.sum(probabilities * np.log2(probabilities))
            return float(entropy)
            
        except Exception as e:
            logger.warning(f"Temporal entropy calculation failed: {e}")
            return 0.0
    
    def _calculate_relational_entropy(self, content_pieces: List[ContentPiece]) -> float:
        """
        Calculate relational entropy based on document relationships and dependencies.
        
        Measures how diverse the relationship patterns are.
        """
        # Extract relationship information
        all_relationships = []
        relationship_types = set()
        
        for piece in content_pieces:
            if piece.relationships:
                all_relationships.extend(piece.relationships)
                relationship_types.update(piece.relationships)
        
        if not relationship_types or len(relationship_types) <= 1:
            return 0.0
        
        try:
            # Count relationship type frequencies
            relationship_counts = Counter(all_relationships)
            
            # Convert to probabilities
            total_relationships = sum(relationship_counts.values())
            probabilities = np.array([
                count / total_relationships 
                for count in relationship_counts.values()
            ])
            
            # Calculate Shannon entropy
            entropy = -np.sum(probabilities * np.log2(probabilities))
            return float(entropy)
            
        except Exception as e:
            logger.warning(f"Relational entropy calculation failed: {e}")
            return 0.0
    
    def _calculate_structural_entropy(self, content_pieces: List[ContentPiece]) -> float:
        """
        Calculate structural entropy based on format and organization patterns.
        
        Measures diversity in content types, sources, and structural characteristics.
        """
        if len(content_pieces) <= 1:
            return 0.0
        
        try:
            # Collect structural features
            features = defaultdict(list)
            
            for piece in content_pieces:
                # Content type distribution
                if piece.content_type:
                    features['content_type'].append(piece.content_type)
                
                # Source distribution
                if piece.source:
                    features['source'].append(piece.source)
                
                # Length patterns (categorized)
                length_category = self._categorize_length(len(piece.text))
                features['length_category'].append(length_category)
                
                # Text structure patterns
                structure_pattern = self._analyze_text_structure(piece.text)
                features['structure_pattern'].append(structure_pattern)
            
            # Calculate entropy for each feature type
            feature_entropies = []
            
            for feature_type, values in features.items():
                if len(set(values)) > 1:  # Only if there's diversity
                    value_counts = Counter(values)
                    total_count = sum(value_counts.values())
                    probabilities = np.array([
                        count / total_count 
                        for count in value_counts.values()
                    ])
                    
                    feature_entropy = -np.sum(probabilities * np.log2(probabilities))
                    feature_entropies.append(feature_entropy)
            
            # Return average entropy across structural features
            if feature_entropies:
                return float(np.mean(feature_entropies))
            else:
                return 0.0
                
        except Exception as e:
            logger.warning(f"Structural entropy calculation failed: {e}")
            return 0.0
    
    def _categorize_length(self, length: int) -> str:
        """Categorize text length into bins."""
        if length < 50:
            return 'very_short'
        elif length < 200:
            return 'short'
        elif length < 800:
            return 'medium'
        elif length < 2000:
            return 'long'
        else:
            return 'very_long'
    
    def _analyze_text_structure(self, text: str) -> str:
        """Analyze basic text structure patterns."""
        # Count structural elements
        lines = text.split('\n')
        sentences = text.split('.')
        
        # Categorize based on structure
        if len(lines) > 10:
            return 'multi_line'
        elif len(sentences) > 5:
            return 'multi_sentence'
        elif '\n' in text:
            return 'line_breaks'
        elif '.' in text:
            return 'sentences'
        else:
            return 'simple'
    
    def _calculate_multidimensional_confidence(self, 
                                             content_pieces: List[ContentPiece],
                                             dimension_entropies: Dict[str, float]) -> float:
        """
        Calculate confidence in multi-dimensional entropy measurement.
        
        Higher confidence when:
        - More metadata is available
        - Multiple dimensions contribute meaningfully
        - Sufficient data points for analysis
        """
        base_confidence = 0.7
        
        # Data availability factor
        total_pieces = len(content_pieces)
        data_factor = min(total_pieces / 10, 1.0)  # Optimal around 10+ pieces
        
        # Metadata completeness factor
        metadata_scores = []
        for piece in content_pieces:
            score = 0
            if piece.timestamp:
                score += 0.25
            if piece.source:
                score += 0.25
            if piece.content_type:
                score += 0.25
            if piece.relationships:
                score += 0.25
            metadata_scores.append(score)
        
        metadata_factor = np.mean(metadata_scores) if metadata_scores else 0.1
        
        # Dimension diversity factor (how many dimensions contribute)
        active_dimensions = sum(1 for entropy in dimension_entropies.values() if entropy > 0.1)
        diversity_factor = active_dimensions / len(dimension_entropies)
        
        # Combine factors
        confidence = base_confidence * data_factor * metadata_factor * diversity_factor
        return max(0.1, min(1.0, confidence))
    
    def adapt_weights(self, 
                     training_data: List[Tuple[List[ContentPiece], float]],
                     learning_rate: float = 0.01) -> Dict[str, float]:
        """
        Adapt dimension weights based on training data.
        
        Args:
            training_data: List of (content_pieces, target_entropy) pairs
            learning_rate: Learning rate for weight adaptation
            
        Returns:
            Updated dimension weights
        """
        if not training_data:
            return self.dimension_weights
        
        # Simple gradient descent on weights
        current_weights = self.dimension_weights.copy()
        
        for content_pieces, target_entropy in training_data:
            # Calculate current prediction
            result = self.calculate_multidimensional_entropy(content_pieces, current_weights)
            error = result.total_entropy - target_entropy
            
            # Update weights (simplified gradient descent)
            for dim in current_weights:
                if dim in result.dimension_entropies:
                    gradient = error * result.dimension_entropies[dim]
                    current_weights[dim] -= learning_rate * gradient
        
        # Normalize weights to sum to 1
        total_weight = sum(current_weights.values())
        if total_weight > 0:
            current_weights = {
                dim: weight / total_weight 
                for dim, weight in current_weights.items()
            }
        
        self.dimension_weights = current_weights
        return current_weights
    
    def get_dimension_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about available entropy dimensions.
        
        Returns:
            Dictionary with dimension information and use cases
        """
        return {
            'semantic': {
                'description': 'Content meaning and topic distribution',
                'measures': 'Diversity in content topics and concepts',
                'best_for': 'Text with varied subjects and themes',
                'weight': self.dimension_weights['semantic']
            },
            'temporal': {
                'description': 'Time-based patterns and distribution',
                'measures': 'How content is spread across time periods',
                'best_for': 'Time-stamped content with temporal patterns',
                'weight': self.dimension_weights['temporal']
            },
            'relational': {
                'description': 'Document relationships and dependencies',
                'measures': 'Diversity in how documents relate to each other',
                'best_for': 'Content with explicit relationships or citations',
                'weight': self.dimension_weights['relational']
            },
            'structural': {
                'description': 'Format and organization patterns',
                'measures': 'Diversity in content types, sources, and structure',
                'best_for': 'Mixed content from different sources and formats',
                'weight': self.dimension_weights['structural']
            }
        }


def content_pieces_from_strings(texts: List[str], 
                               timestamps: Optional[List[datetime]] = None,
                               sources: Optional[List[str]] = None) -> List[ContentPiece]:
    """
    Utility function to create ContentPiece objects from simple text lists.
    
    Args:
        texts: List of text strings
        timestamps: Optional timestamps for each text
        sources: Optional source identifiers
        
    Returns:
        List of ContentPiece objects
    """
    pieces = []
    
    for i, text in enumerate(texts):
        timestamp = timestamps[i] if timestamps and i < len(timestamps) else None
        source = sources[i] if sources and i < len(sources) else None
        
        pieces.append(ContentPiece(
            text=text,
            timestamp=timestamp,
            source=source
        ))
    
    return pieces