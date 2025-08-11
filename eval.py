import json
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Union
import warnings
import logging
from dataclasses import dataclass
from scipy import stats
from scipy.stats import spearmanr, pearsonr, kendalltau
from scipy.interpolate import make_interp_spline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, cmudict
from nltk.tag import pos_tag
from nltk.tree import Tree
from nltk.chunk import ne_chunk
import string
import math
import re
from tqdm import tqdm
import argparse
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'dof_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

# Download required NLTK data with comprehensive error handling
def download_nltk_resources():
    """Download all required NLTK resources with robust error handling."""
    required_resources = {
        'tokenizers': ['punkt', 'punkt_tab'],
        'taggers': ['averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng'],
        'corpora': ['stopwords', 'cmudict', 'words', 'brown'],
        'chunkers': ['maxent_ne_chunker'],
        'misc': ['perluniprops', 'nonbreaking_prefixes']
    }
    
    for category, resources in required_resources.items():
        for resource in resources:
            try:
                if category == 'tokenizers':
                    nltk.data.find(f'tokenizers/{resource}')
                elif category == 'taggers':
                    nltk.data.find(f'taggers/{resource}')
                elif category == 'corpora':
                    nltk.data.find(f'corpora/{resource}')
                elif category == 'chunkers':
                    nltk.data.find(f'chunkers/{resource}')
                else:
                    nltk.data.find(resource)
            except LookupError:
                try:
                    logger.info(f"Downloading NLTK {category}: {resource}")
                    nltk.download(resource, quiet=True)
                except Exception as e:
                    logger.warning(f"Could not download {resource}: {e}")
                    # Try alternative download method
                    try:
                        if resource == 'averaged_perceptron_tagger_eng':
                            nltk.download('averaged_perceptron_tagger', quiet=True)
                        elif resource == 'punkt_tab':
                            nltk.download('punkt', quiet=True)
                    except:
                        pass

# Download all required NLTK resources
download_nltk_resources()

# Initialize CMU Pronouncing Dictionary for syllable counting
try:
    CMU_DICT = cmudict.dict()
except Exception as e:
    logger.warning(f"Could not load CMU dictionary: {e}")
    try:
        nltk.download('cmudict', quiet=True)
        CMU_DICT = cmudict.dict()
    except:
        CMU_DICT = {}

# Initialize stopwords
try:
    STOPWORDS = set(stopwords.words('english'))
except Exception as e:
    logger.warning(f"Could not load stopwords: {e}")
    try:
        nltk.download('stopwords', quiet=True)
        STOPWORDS = set(stopwords.words('english'))
    except:
        # Minimal fallback stopwords
        STOPWORDS = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been'}


@dataclass
class MetricResult:
    """Data class for storing metric results with metadata"""
    value: float
    std: float
    ci_lower: float
    ci_upper: float
    n_samples: int
    
    
class AdvancedTextMetricsCalculator:
    """
    Advanced text metrics calculator implementing state-of-the-art 
    lexical diversity and readability measures.
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize the calculator with configuration parameters.
        
        Args:
            confidence_level: Confidence level for statistical calculations
        """
        self.confidence_level = confidence_level
        self.z_score = stats.norm.ppf((1 + confidence_level) / 2)
        self.punctuation = set(string.punctuation)
        self._initialize_pos_weights()
        
    def _initialize_pos_weights(self):
        """Initialize POS tag weights for content word identification"""
        self.content_pos_tags = {
            'NN', 'NNS', 'NNP', 'NNPS',  # Nouns
            'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',  # Verbs
            'JJ', 'JJR', 'JJS',  # Adjectives
            'RB', 'RBR', 'RBS',  # Adverbs
        }
        
    def calculate_mtld(self, tokens: List[str], threshold: float = 0.72) -> float:
        """
        Calculate Measure of Textual Lexical Diversity (MTLD).
        
        Implementation based on McCarthy & Jarvis (2010).
        "MTLD, vocd-D, and HD-D: A validation study of sophisticated 
        approaches to lexical diversity assessment"
        
        Args:
            tokens: List of tokens
            threshold: TTR threshold for factor completion
            
        Returns:
            MTLD score
        """
        # For very short texts, use alternative measure
        if len(tokens) < 30:
            # Use Root TTR for short texts
            types = len(set([t.lower() for t in tokens]))
            return types / math.sqrt(len(tokens)) if tokens else 0.0
        
        def _compute_forward_mtld(tokens: List[str], threshold: float) -> float:
            """Compute forward MTLD with improved partial factor calculation"""
            factors = 0.0
            factor_length = 0
            types_seen = set()
            
            for token in tokens:
                factor_length += 1
                types_seen.add(token.lower())
                
                ttr = len(types_seen) / factor_length
                
                if ttr <= threshold:
                    factors += 1
                    factor_length = 0
                    types_seen = set()
            
            # Improved partial factor calculation
            if factor_length > 0:
                ttr = len(types_seen) / factor_length
                # Only add partial factor if we have enough tokens
                if factor_length >= 10:
                    partial_factor = (1 - ttr) / (1 - threshold)
                    # Clamp partial factor between 0 and 1
                    partial_factor = max(0, min(1, partial_factor))
                    factors += partial_factor
                else:
                    # For very small remaining segments, use proportional factor
                    factors += factor_length / 30.0  # Assume 30 tokens as minimum factor length
            
            # Ensure we have at least some factor count
            if factors == 0:
                factors = 0.1
            
            return len(tokens) / factors
        
        # Bidirectional MTLD for robustness
        forward_mtld = _compute_forward_mtld(tokens, threshold)
        backward_mtld = _compute_forward_mtld(tokens[::-1], threshold)
        
        # Check for large discrepancy
        if forward_mtld > 0 and backward_mtld > 0:
            discrepancy_ratio = abs(forward_mtld - backward_mtld) / max(forward_mtld, backward_mtld)
            
            if discrepancy_ratio > 0.5:
                # Large discrepancy - use more robust measure
                logger.debug(f"MTLD discrepancy: forward={forward_mtld:.2f}, backward={backward_mtld:.2f}")
                
                # Use geometric mean for large discrepancies (more robust to outliers)
                return math.sqrt(forward_mtld * backward_mtld)
            elif discrepancy_ratio > 0.3:
                # Moderate discrepancy - use weighted average favoring the smaller value
                min_val = min(forward_mtld, backward_mtld)
                max_val = max(forward_mtld, backward_mtld)
                # Weight towards the more conservative estimate
                return (2 * min_val + max_val) / 3
        
        # Normal case - simple average
        return (forward_mtld + backward_mtld) / 2
    
    def calculate_hdd(self, tokens: List[str], sample_size: int = 42) -> float:
        """
        Calculate HD-D (Hypergeometric Distribution Diversity).
        
        Based on McCarthy & Jarvis (2007).
        "vocd: A theoretical and empirical evaluation"
        
        Args:
            tokens: List of tokens
            sample_size: Standard sample size for HD-D
            
        Returns:
            HD-D score
        """
        token_count = len(tokens)
        
        # For short texts, use alternative diversity measure
        if token_count < sample_size:
            # Use Simpson's diversity for short texts
            return self.calculate_simpson_diversity(tokens) * 100  # Scale to similar range
        
        type_count = len(set([t.lower() for t in tokens]))
        
        # Adaptive sample size range based on text length
        min_sample = max(35, int(token_count * 0.3))
        max_sample = min(50, int(token_count * 0.8))
        
        if min_sample >= max_sample:
            min_sample = max(35, token_count // 2)
            max_sample = min(min_sample + 10, token_count)
        
        # Calculate hypergeometric probabilities
        contributions = []
        
        for sample in range(min_sample, max_sample + 1):
            if sample > token_count:
                break
                
            # Expected TTR for this sample size
            expected_ttr = 0
            
            # Limit the range to avoid computational issues
            max_types_in_sample = min(type_count, sample)
            
            for num_types in range(1, max_types_in_sample + 1):
                # Hypergeometric probability
                prob = self._hypergeometric_probability(
                    token_count, type_count, sample, num_types
                )
                if prob > 0:
                    expected_ttr += prob * (num_types / sample)
            
            # Contribution to HD-D
            if expected_ttr < 1:
                contribution = type_count * (1 - expected_ttr)
                contributions.append(contribution)
        
        # Return mean contribution, or fallback value
        if contributions:
            return np.mean(contributions)
        else:
            # Fallback to simple diversity measure
            return type_count / math.sqrt(token_count) if token_count > 0 else 0.0
    
    def _hypergeometric_probability(self, N: int, K: int, n: int, k: int) -> float:
        """
        Calculate hypergeometric probability.
        
        P(X = k) = C(K, k) * C(N-K, n-k) / C(N, n)
        """
        from math import comb
        
        if k > min(n, K) or k < max(0, n - N + K):
            return 0.0
        
        try:
            return (comb(K, k) * comb(N - K, n - k)) / comb(N, n)
        except (ValueError, OverflowError):
            return 0.0
    
    def calculate_shannon_entropy(self, tokens: List[str], normalize: bool = True) -> float:
        """
        Calculate Shannon entropy with optional normalization.
        
        H(X) = -Σ p(x) * log2(p(x))
        
        Args:
            tokens: List of tokens
            normalize: Whether to normalize by max entropy
            
        Returns:
            Shannon entropy value
        """
        if not tokens:
            return 0.0
        
        # Calculate token frequencies
        freq_dist = Counter([t.lower() for t in tokens])
        total = sum(freq_dist.values())
        
        # Calculate entropy
        entropy = 0.0
        for count in freq_dist.values():
            if count > 0:
                prob = count / total
                entropy -= prob * math.log2(prob)
        
        # Normalize by maximum possible entropy
        if normalize and len(freq_dist) > 1:
            max_entropy = math.log2(len(freq_dist))
            entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        return entropy
    
    def calculate_simpson_diversity(self, tokens: List[str]) -> float:
        """
        Calculate Simpson's Diversity Index.
        
        D = 1 - Σ(n(n-1))/(N(N-1))
        """
        if len(tokens) < 2:
            return 0.0
        
        freq_dist = Counter([t.lower() for t in tokens])
        N = len(tokens)
        
        sum_n_n_minus_1 = sum(n * (n - 1) for n in freq_dist.values())
        
        return 1 - (sum_n_n_minus_1 / (N * (N - 1)))
    
    def calculate_yule_k(self, tokens: List[str]) -> float:
        """
        Calculate Yule's K characteristic.
        
        K = 10^4 * (M2 - N) / N^2
        where M2 = Σ r^2 * V(r)
        """
        if not tokens:
            return 0.0
        
        freq_dist = Counter([t.lower() for t in tokens])
        freq_spectrum = Counter(freq_dist.values())
        
        N = len(tokens)
        M2 = sum(r * r * freq_spectrum[r] for r in freq_spectrum)
        
        if N == 0:
            return 0.0
        
        return 10000 * (M2 - N) / (N * N)
    
    def calculate_mean_sentence_length(self, text: str) -> Tuple[float, float]:
        """
        Calculate mean sentence length with standard deviation.
        
        Returns:
            Tuple of (mean, std)
        """
        sentences = sent_tokenize(text)
        if not sentences:
            return 0.0, 0.0
        
        lengths = [len(word_tokenize(sent)) for sent in sentences]
        
        return np.mean(lengths), np.std(lengths)
    
    def calculate_syntactic_complexity(self, text: str) -> Dict[str, float]:
        """
        Calculate multiple syntactic complexity metrics.
        
        Returns:
            Dictionary with various complexity measures
        """
        sentences = sent_tokenize(text)
        
        metrics = {
            'mean_sentence_length': 0.0,
            'std_sentence_length': 0.0,
            'mean_clause_length': 0.0,
            'subordination_index': 0.0,
            'coordination_index': 0.0,
            'mean_parse_tree_depth': 0.0
        }
        
        if not sentences:
            return metrics
        
        sentence_lengths = []
        clause_lengths = []
        subordinate_clauses = 0
        coordinate_clauses = 0
        
        for sent in sentences:
            tokens = word_tokenize(sent)
            sentence_lengths.append(len(tokens))
            
            try:
                # POS tagging for clause identification
                pos_tags = pos_tag(tokens)
                
                # Count subordinate conjunctions
                sub_conj = ['because', 'although', 'while', 'since', 'if', 'when', 'whereas']
                coord_conj = ['and', 'but', 'or', 'nor', 'for', 'yet', 'so']
                
                for token, tag in pos_tags:
                    if token.lower() in sub_conj:
                        subordinate_clauses += 1
                    elif token.lower() in coord_conj:
                        coordinate_clauses += 1
            except Exception as e:
                logger.debug(f"POS tagging failed in syntactic complexity: {e}")
                # Simple fallback without POS tagging
                for token in tokens:
                    if token.lower() in ['because', 'although', 'while', 'since', 'if', 'when', 'whereas']:
                        subordinate_clauses += 1
                    elif token.lower() in ['and', 'but', 'or', 'nor', 'for', 'yet', 'so']:
                        coordinate_clauses += 1
        
        metrics['mean_sentence_length'] = np.mean(sentence_lengths)
        metrics['std_sentence_length'] = np.std(sentence_lengths)
        metrics['subordination_index'] = subordinate_clauses / len(sentences)
        metrics['coordination_index'] = coordinate_clauses / len(sentences)
        
        return metrics
    
    def calculate_lexical_density(self, tokens: List[str]) -> float:
        """
        Calculate lexical density using POS-based content word identification.
        
        LD = (Number of content words / Total words) × 100
        """
        if not tokens:
            return 0.0
        
        try:
            # POS tagging
            pos_tags = pos_tag(tokens)
            
            # Count content words
            content_words = sum(1 for token, tag in pos_tags 
                              if tag in self.content_pos_tags and 
                              token not in self.punctuation)
            
            # Filter out punctuation for total count
            total_words = sum(1 for token in tokens if token not in self.punctuation)
            
            return (content_words / total_words) if total_words > 0 else 0.0
        except Exception as e:
            # Fallback: simple function word list approach
            logger.debug(f"POS tagging failed, using fallback for lexical density: {e}")
            
            function_words = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
                'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                'could', 'should', 'may', 'might', 'must', 'can', 'shall', 'it',
                'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'we', 'they',
                'what', 'which', 'who', 'when', 'where', 'why', 'how', 'not', 'no',
                'nor', 'so', 'than', 'too', 'very', 'just', 'also'
            }
            
            tokens_lower = [t.lower() for t in tokens if t not in self.punctuation]
            content_words = [t for t in tokens_lower if t not in function_words]
            
            return len(content_words) / len(tokens_lower) if tokens_lower else 0.0
    
    def calculate_lexical_sophistication(self, tokens: List[str]) -> Dict[str, float]:
        """
        Calculate lexical sophistication metrics.
        
        Returns:
            Dictionary with sophistication measures
        """
        if not tokens:
            return {'rare_word_ratio': 0.0, 'avg_word_length': 0.0}
        
        # Word frequency analysis (simplified - in production, use corpus frequencies)
        word_lengths = [len(token) for token in tokens if token not in self.punctuation]
        
        # Consider words > 8 characters as sophisticated
        sophisticated_words = sum(1 for length in word_lengths if length > 8)
        
        return {
            'rare_word_ratio': sophisticated_words / len(word_lengths) if word_lengths else 0.0,
            'avg_word_length': np.mean(word_lengths) if word_lengths else 0.0
        }
    
    def calculate_flesch_reading_ease(self, text: str) -> float:
        """
        Calculate Flesch Reading Ease score with improved syllable counting.
        
        FRE = 206.835 - 1.015(total words/total sentences) - 84.6(total syllables/total words)
        """
        sentences = sent_tokenize(text)
        if not sentences:
            return 0.0
        
        words = word_tokenize(text)
        words = [w for w in words if w not in self.punctuation and w.isalpha()]
        
        if not words or not sentences:
            return 0.0
        
        # Count syllables using CMU dictionary with fallback
        total_syllables = sum(self._count_syllables(word) for word in words)
        
        avg_sentence_length = len(words) / len(sentences)
        avg_syllables_per_word = total_syllables / len(words)
        
        fre = 206.835 - 1.015 * avg_sentence_length - 84.6 * avg_syllables_per_word
        
        return max(0, min(100, fre))
    
    def calculate_flesch_kincaid_grade(self, text: str) -> float:
        """
        Calculate Flesch-Kincaid Grade Level.
        
        FKGL = 0.39(total words/total sentences) + 11.8(total syllables/total words) - 15.59
        """
        sentences = sent_tokenize(text)
        if not sentences:
            return 0.0
        
        words = word_tokenize(text)
        words = [w for w in words if w not in self.punctuation and w.isalpha()]
        
        if not words or not sentences:
            return 0.0
        
        total_syllables = sum(self._count_syllables(word) for word in words)
        
        avg_sentence_length = len(words) / len(sentences)
        avg_syllables_per_word = total_syllables / len(words)
        
        fkgl = 0.39 * avg_sentence_length + 11.8 * avg_syllables_per_word - 15.59
        
        return max(0, fkgl)
    
    def _count_syllables(self, word: str) -> int:
        """
        Count syllables using CMU dictionary with fallback heuristic.
        """
        word_lower = word.lower()
        
        # Try CMU dictionary first
        if word_lower in CMU_DICT:
            # CMU dict returns list of pronunciations
            # Count stressed vowels (numbers in pronunciation)
            pronunciation = CMU_DICT[word_lower][0]
            return len([p for p in pronunciation if p[-1].isdigit()])
        
        # Fallback: heuristic syllable counting
        vowels = 'aeiouAEIOU'
        syllable_count = 0
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel
        
        # Adjust for silent e
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        # Ensure at least one syllable
        return max(1, syllable_count)
    
    def calculate_all_metrics(self, text: str) -> Dict[str, float]:
        """
        Calculate all core and extended metrics.
        
        Returns:
            Comprehensive dictionary of all metrics
        """
        tokens = word_tokenize(text)
        tokens_no_punct = [t for t in tokens if t not in self.punctuation]
        
        # Core metrics
        metrics = {
            'mtld': self.calculate_mtld(tokens_no_punct),
            'hdd': self.calculate_hdd(tokens_no_punct),
            'shannon_entropy': self.calculate_shannon_entropy(tokens_no_punct),
            'lexical_density': self.calculate_lexical_density(tokens),
            'flesch_reading_ease': self.calculate_flesch_reading_ease(text),
        }
        
        # Sentence length metrics
        mean_sent_len, std_sent_len = self.calculate_mean_sentence_length(text)
        metrics['mean_sentence_length'] = mean_sent_len
        metrics['std_sentence_length'] = std_sent_len
        
        # Extended metrics for comprehensive analysis
        metrics['simpson_diversity'] = self.calculate_simpson_diversity(tokens_no_punct)
        metrics['yule_k'] = self.calculate_yule_k(tokens_no_punct)
        metrics['flesch_kincaid_grade'] = self.calculate_flesch_kincaid_grade(text)
        
        # Syntactic complexity
        syntactic_metrics = self.calculate_syntactic_complexity(text)
        metrics.update({f'syntactic_{k}': v for k, v in syntactic_metrics.items()})
        
        # Lexical sophistication
        sophistication_metrics = self.calculate_lexical_sophistication(tokens_no_punct)
        metrics.update({f'lexical_{k}': v for k, v in sophistication_metrics.items()})
        
        # Type-Token Ratio variants
        metrics['ttr'] = len(set(tokens_no_punct)) / len(tokens_no_punct) if tokens_no_punct else 0
        metrics['root_ttr'] = len(set(tokens_no_punct)) / math.sqrt(len(tokens_no_punct)) if tokens_no_punct else 0
        metrics['log_ttr'] = len(set(tokens_no_punct)) / math.log(len(tokens_no_punct)) if len(tokens_no_punct) > 1 else 0
        
        return metrics


class StatisticalAnalyzer:
    """
    Statistical analysis toolkit for DoF experiments.
    """
    
    @staticmethod
    def perform_correlation_analysis(df: pd.DataFrame, 
                                    dof_column: str = 'dof_value') -> pd.DataFrame:
        """
        Perform comprehensive correlation analysis.
        """
        metrics = ['mtld', 'hdd', 'shannon_entropy', 'mean_sentence_length', 
                  'lexical_density', 'flesch_reading_ease']
        
        results = []
        
        for metric in metrics:
            if f'{metric}_mean' in df.columns:
                # Pearson correlation
                pearson_r, pearson_p = pearsonr(df[dof_column], df[f'{metric}_mean'])
                
                # Spearman correlation
                spearman_rho, spearman_p = spearmanr(df[dof_column], df[f'{metric}_mean'])
                
                # Kendall's tau
                kendall_tau, kendall_p = kendalltau(df[dof_column], df[f'{metric}_mean'])
                
                results.append({
                    'metric': metric,
                    'pearson_r': pearson_r,
                    'pearson_p': pearson_p,
                    'spearman_rho': spearman_rho,
                    'spearman_p': spearman_p,
                    'kendall_tau': kendall_tau,
                    'kendall_p': kendall_p
                })
        
        return pd.DataFrame(results)
    
    @staticmethod
    def perform_regression_analysis(df: pd.DataFrame, 
                                   dof_column: str = 'dof_value') -> Dict:
        """
        Perform polynomial regression analysis for trend identification.
        """
        metrics = ['mtld', 'hdd', 'shannon_entropy', 'mean_sentence_length', 
                  'lexical_density', 'flesch_reading_ease']
        
        regression_results = {}
        
        for metric in metrics:
            if f'{metric}_mean' not in df.columns:
                continue
            
            X = df[dof_column].values.reshape(-1, 1)
            y = df[f'{metric}_mean'].values
            
            # Test polynomial degrees 1-3
            results = {}
            for degree in range(1, 4):
                # Create polynomial features
                from sklearn.preprocessing import PolynomialFeatures
                poly = PolynomialFeatures(degree=degree)
                X_poly = poly.fit_transform(X)
                
                # Fit model
                model = LinearRegression()
                model.fit(X_poly, y)
                y_pred = model.predict(X_poly)
                
                # Calculate metrics
                r2 = r2_score(y, y_pred)
                aic = len(y) * np.log(np.sum((y - y_pred) ** 2) / len(y)) + 2 * (degree + 1)
                bic = len(y) * np.log(np.sum((y - y_pred) ** 2) / len(y)) + np.log(len(y)) * (degree + 1)
                
                results[degree] = {
                    'r2': r2,
                    'aic': aic,
                    'bic': bic,
                    'coefficients': model.coef_.tolist(),
                    'intercept': model.intercept_
                }
            
            # Select best model based on AIC
            best_degree = min(results.keys(), key=lambda x: results[x]['aic'])
            regression_results[metric] = {
                'best_degree': best_degree,
                'models': results
            }
        
        return regression_results
    
    @staticmethod
    def perform_anova(df: pd.DataFrame, group_column: str = 'dof_value') -> pd.DataFrame:
        """
        Perform one-way ANOVA for each metric across DoF values.
        """
        metrics = ['mtld', 'hdd', 'shannon_entropy', 'mean_sentence_length', 
                  'lexical_density', 'flesch_reading_ease']
        
        results = []
        
        for metric in metrics:
            if metric not in df.columns:
                continue
            
            # Group data by DoF value
            groups = [group[metric].dropna().values 
                     for name, group in df.groupby(group_column)]
            
            if len(groups) < 2:
                continue
            
            # Perform ANOVA
            f_stat, p_value = stats.f_oneway(*groups)
            
            # Effect size (eta-squared)
            grand_mean = df[metric].mean()
            ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
            ss_total = sum((val - grand_mean) ** 2 for g in groups for val in g)
            eta_squared = ss_between / ss_total if ss_total > 0 else 0
            
            results.append({
                'metric': metric,
                'f_statistic': f_stat,
                'p_value': p_value,
                'eta_squared': eta_squared,
                'interpretation': 'Significant' if p_value < 0.05 else 'Not significant'
            })
        
        return pd.DataFrame(results)


class AdvancedVisualization:
    """
    Publication-quality visualization for DoF analysis.
    """
    
    def __init__(self, style: str = 'seaborn-v0_8-paper'):
        """Initialize with publication style."""
        plt.style.use(style)
        self.colors = sns.color_palette("husl", 10)
        self.figure_dpi = 300
        
    def create_comprehensive_figure(self, aggregated_df: pd.DataFrame, 
                                  correlation_df: pd.DataFrame,
                                  regression_results: Dict,
                                  output_path: str = 'dof_comprehensive_analysis.pdf'):
        """
        Create publication-ready comprehensive figure with all analyses.
        """
        # Create figure with GridSpec for complex layout
        from matplotlib.gridspec import GridSpec
        
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.25)
        
        # Main metrics plots (2x3)
        metrics = ['mtld', 'hdd', 'shannon_entropy', 
                  'mean_sentence_length', 'lexical_density', 'flesch_reading_ease']
        metric_labels = [
            'MTLD\n(Lexical Diversity)',
            'HD-D\n(Hypergeometric Diversity)',
            'Shannon Entropy\n(Information Content)',
            'Mean Sentence Length\n(Syntactic Complexity)',
            'Lexical Density\n(Content Richness)',
            'Flesch Reading Ease\n(Readability)'
        ]
        
        for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = fig.add_subplot(gs[idx // 3, idx % 3])
            self._plot_metric_with_fit(ax, aggregated_df, metric, label, 
                                      regression_results.get(metric, {}))
        
        # Add overall title with metadata
        fig.suptitle('Impact of Degree of Freedom on Language Model Generation Quality', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Save figure
        plt.savefig(output_path, dpi=self.figure_dpi, bbox_inches='tight', 
                   format='pdf', metadata={'Creator': 'DoF Analysis Framework'})
        plt.show()
        
        logger.info(f"Comprehensive figure saved to: {output_path}")
        
    def _plot_metric_with_fit(self, ax, df, metric, label, regression_info):
        """Plot individual metric with regression fit and confidence intervals."""
        
        # Plot data points with error bars
        for model in df['model_name'].unique():
            model_data = df[df['model_name'] == model].sort_values('dof_value')
            
            # Main plot with error bars
            ax.errorbar(model_data['dof_value'], 
                       model_data[f'{metric}_mean'],
                       yerr=model_data[f'{metric}_std'],
                       marker='o', 
                       label=model,
                       capsize=3,
                       capthick=1,
                       linewidth=1.5,
                       markersize=6,
                       alpha=0.8)
            
            # Add regression line if available
            if regression_info and 'models' in regression_info:
                best_degree = regression_info.get('best_degree', 1)
                model_info = regression_info['models'][best_degree]
                
                # Generate smooth curve
                x_smooth = np.linspace(0, 1, 100)
                from sklearn.preprocessing import PolynomialFeatures
                poly = PolynomialFeatures(degree=best_degree)
                X_smooth = poly.fit_transform(x_smooth.reshape(-1, 1))
                
                # Predict using stored coefficients
                y_smooth = X_smooth @ model_info['coefficients'] + model_info['intercept']
                
                ax.plot(x_smooth, y_smooth, '--', alpha=0.5, linewidth=1)
        
        # Formatting
        ax.set_xlabel('Degree of Freedom (DoF)', fontsize=10, fontweight='bold')
        ax.set_ylabel(label, fontsize=10, fontweight='bold')
        ax.set_xlim(-0.05, 1.05)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=8, framealpha=0.9)
        
        # Add R² value if available
        if regression_info and 'models' in regression_info:
            best_degree = regression_info.get('best_degree', 1)
            r2 = regression_info['models'][best_degree]['r2']
            ax.text(0.95, 0.05, f'R² = {r2:.3f}', 
                   transform=ax.transAxes, ha='right', va='bottom',
                   fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def create_correlation_heatmap(self, correlation_df: pd.DataFrame, 
                                  output_path: str = 'dof_correlation_heatmap.pdf'):
        """Create correlation heatmap for all metrics, robust to NaNs."""
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        panels = [
            ("pearson_r", "Pearson's r"),
            ("spearman_rho", "Spearman's ρ"),
            ("kendall_tau", "Kendall's τ"),
        ]
        
        for idx, (col, title) in enumerate(panels):
            # Use only valid rows for this correlation type
            sub = correlation_df[['metric', col]].dropna()
            if sub.empty:
                axes[idx].set_title(title + " (no valid data)", fontweight='bold')
                axes[idx].axis('off')
                continue
            
            metrics_valid = sub['metric'].tolist()
            data = sub[col].to_numpy().reshape(-1, 1)
            
            im = axes[idx].imshow(data, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
            axes[idx].set_yticks(range(len(metrics_valid)))
            axes[idx].set_yticklabels(metrics_valid)
            axes[idx].set_xticks([0])
            axes[idx].set_xticklabels(['DoF'])
            axes[idx].set_title(title, fontweight='bold')
            
            # Add value labels safely
            for i in range(data.shape[0]):
                val = data[i, 0]
                if np.isnan(val):
                    txt = "nan"
                    color = 'black'
                else:
                    txt = f"{val:.3f}"
                    color = 'white' if abs(val) > 0.5 else 'black'
                axes[idx].text(0, i, txt, ha='center', va='center', color=color, fontweight='bold')
            
            cbar = plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
            cbar.ax.set_ylabel('Correlation Coefficient', rotation=270, labelpad=15)
        
        plt.suptitle('Correlation Analysis: DoF vs. Linguistic Metrics', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.figure_dpi, bbox_inches='tight', format='pdf')
        plt.show()
        
        logger.info(f"Correlation heatmap saved to: {output_path}")
    
    def create_distribution_plots(self, df: pd.DataFrame, 
                                output_path: str = 'dof_distributions.pdf'):
        """Create distribution plots for each metric across DoF values."""
        
        metrics = ['mtld', 'hdd', 'shannon_entropy', 
                  'mean_sentence_length', 'lexical_density', 'flesch_reading_ease']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            if metric not in df.columns:
                continue
            
            # Create violin plot for each DoF value
            dof_values = sorted(df['dof_value'].unique())
            data_by_dof = [df[df['dof_value'] == dof][metric].dropna() 
                          for dof in dof_values]
            
            parts = axes[idx].violinplot(data_by_dof, positions=dof_values, 
                                        widths=0.1, showmeans=True, showmedians=True)
            
            # Customize colors
            for pc in parts['bodies']:
                pc.set_facecolor(self.colors[idx])
                pc.set_alpha(0.7)
            
            axes[idx].set_xlabel('DoF Value', fontweight='bold')
            axes[idx].set_ylabel(metric.replace('_', ' ').title(), fontweight='bold')
            axes[idx].set_title(f'Distribution of {metric.replace("_", " ").title()}')
            axes[idx].grid(True, alpha=0.3)
        
        plt.suptitle('Metric Distributions Across DoF Values', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.figure_dpi, bbox_inches='tight', format='pdf')
        plt.show()
        
        logger.info(f"Distribution plots saved to: {output_path}")


class DataProcessor:
    """
    Comprehensive data processing pipeline for DoF experiments.
    """
    
    def __init__(self, calculator: AdvancedTextMetricsCalculator):
        """
        Initialize with metrics calculator.
        
        Args:
            calculator: Instance of AdvancedTextMetricsCalculator
        """
        self.calculator = calculator
        
    def load_json_files(self, folder_path: str) -> List[Dict]:
        """
        Load all JSON files from specified folder with validation.
        
        Args:
            folder_path: Path to folder containing JSON files
            
        Returns:
            List of loaded JSON data
        """
        json_files = glob.glob(os.path.join(folder_path, '*.json'))
        all_data = []
        logger.info(f"Found {len(json_files)} JSON files in {folder_path}")
        
        for file_path in tqdm(json_files, desc="Loading JSON files"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # Validate data structure
                    if self._validate_json_structure(data):
                        all_data.append(data)
                        logger.debug(f"Successfully loaded: {os.path.basename(file_path)}")
                    else:
                        logger.warning(f"Invalid structure in: {os.path.basename(file_path)}")
                        
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error in {file_path}: {e}")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        
        logger.info(f"Successfully loaded {len(all_data)} valid JSON files")
        return all_data
    
    def _validate_json_structure(self, data: Dict) -> bool:
        """Validate JSON data structure."""
        required_fields = ['dof_value', 'model_name', 'results']
        
        for field in required_fields:
            if field not in data:
                return False
        
        if not isinstance(data['results'], list):
            return False
        
        return True
    
    def process_data(self, all_data: List[Dict], 
                    sample_limit: Optional[int] = None) -> pd.DataFrame:
        """
        Process data and calculate all metrics with progress tracking.
        
        Args:
            all_data: List of JSON data
            sample_limit: Optional limit on samples per file
            
        Returns:
            DataFrame with calculated metrics
        """
        results = []
        total_samples = sum(len(d.get('results', [])) for d in all_data)
        
        with tqdm(total=total_samples, desc="Processing samples") as pbar:
            for data in all_data:
                dof_value = data.get('dof_value', 0)
                model_name = data.get('model_name', 'unknown')
                dataset = data.get('dataset', 'unknown')
                timestamp = data.get('timestamp', 'unknown')
                
                samples = data.get('results', [])
                if sample_limit:
                    samples = samples[:sample_limit]
                
                for result in samples:
                    pbar.update(1)
                    
                    if result.get('error') is not None:
                        continue
                    
                    generated_text = result.get('generated_continuation', '')
                    
                    if not generated_text or len(generated_text.strip()) < 10:
                        continue
                    
                    try:
                        # Calculate all metrics
                        metrics = self.calculator.calculate_all_metrics(generated_text)
                        
                        # Validate metrics - filter out unrealistic values
                        valid_metrics = True
                        for key, value in metrics.items():
                            if isinstance(value, (int, float)):
                                # Check for NaN or Inf
                                if np.isnan(value) or np.isinf(value):
                                    logger.debug(f"Invalid metric {key}: {value}")
                                    valid_metrics = False
                                    break
                                # Check for unrealistic MTLD values
                                if key == 'mtld' and value > 500:
                                    logger.debug(f"Unrealistic MTLD value: {value}")
                                    metrics['mtld'] = min(value, 200)  # Cap at reasonable maximum
                                # Check for unrealistic HD-D values  
                                if key == 'hdd' and value > 200:
                                    logger.debug(f"Unrealistic HD-D value: {value}")
                                    metrics['hdd'] = min(value, 150)  # Cap at reasonable maximum
                        
                        if not valid_metrics:
                            continue
                        
                        # Create result row
                        row = {
                            'dof_value': dof_value,
                            'model_name': model_name,
                            'dataset': dataset,
                            'timestamp': timestamp,
                            'index': result.get('index', -1),
                            'text_length': len(generated_text),
                            **metrics
                        }
                        results.append(row)
                        
                    except Exception as e:
                        logger.warning(f"Error processing sample {result.get('index', 'unknown')}: {e}")
        
        df = pd.DataFrame(results)
        logger.info(f"Processed {len(df)} valid samples")
        
        return df
    
    def aggregate_metrics(self, df: pd.DataFrame, 
                        confidence_level: float = 0.95) -> pd.DataFrame:
        """
        Aggregate metrics by DoF value with confidence intervals.
        
        Args:
            df: DataFrame with raw metrics
            confidence_level: Confidence level for intervals
            
        Returns:
            Aggregated DataFrame with statistics
        """
        # Core metrics to aggregate
        metrics = ['mtld', 'hdd', 'shannon_entropy', 'mean_sentence_length', 
                  'lexical_density', 'flesch_reading_ease', 'simpson_diversity',
                  'yule_k', 'flesch_kincaid_grade', 'ttr', 'root_ttr', 'log_ttr']
        
        # Calculate z-score for confidence intervals
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        
        aggregated_data = []
        
        for (dof_value, model_name), group in df.groupby(['dof_value', 'model_name']):
            row = {
                'dof_value': dof_value,
                'model_name': model_name,
                'n_samples': len(group)
            }
            
            for metric in metrics:
                if metric in group.columns:
                    values = group[metric].dropna()
                    
                    if len(values) > 0:
                        mean_val = values.mean()
                        std_val = values.std()
                        sem = std_val / np.sqrt(len(values))
                        
                        row[f'{metric}_mean'] = mean_val
                        row[f'{metric}_std'] = std_val
                        row[f'{metric}_sem'] = sem
                        row[f'{metric}_ci_lower'] = mean_val - z_score * sem
                        row[f'{metric}_ci_upper'] = mean_val + z_score * sem
                        row[f'{metric}_median'] = values.median()
                        row[f'{metric}_iqr'] = values.quantile(0.75) - values.quantile(0.25)
            
            aggregated_data.append(row)
        
        aggregated_df = pd.DataFrame(aggregated_data)
        
        # Sort by DoF value for better visualization
        aggregated_df = aggregated_df.sort_values(['model_name', 'dof_value'])
        
        logger.info(f"Aggregated metrics for {len(aggregated_df)} DoF-model combinations")
        
        return aggregated_df
    
    def perform_outlier_detection(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and mark outliers using IQR method.
        
        Args:
            df: DataFrame with metrics
            
        Returns:
            DataFrame with outlier flags
        """
        metrics = ['mtld', 'hdd', 'shannon_entropy', 'mean_sentence_length', 
                  'lexical_density', 'flesch_reading_ease']
        
        df = df.copy()
        
        for metric in metrics:
            if metric not in df.columns:
                continue
            
            Q1 = df[metric].quantile(0.25)
            Q3 = df[metric].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            df[f'{metric}_outlier'] = (df[metric] < lower_bound) | (df[metric] > upper_bound)
        
        outlier_count = sum(df[[col for col in df.columns if '_outlier' in col]].any(axis=1))
        logger.info(f"Detected {outlier_count} samples with at least one outlier metric")
        
        return df


class ReportGenerator:
    """
    Generate comprehensive analysis reports.
    """
    
    def __init__(self, output_dir: str = 'dof_analysis_results'):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory for output files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def generate_latex_table(self, aggregated_df: pd.DataFrame, 
                            correlation_df: pd.DataFrame) -> str:
        """
        Generate LaTeX table for publication.
        
        Args:
            aggregated_df: Aggregated metrics DataFrame
            correlation_df: Correlation analysis DataFrame
            
        Returns:
            LaTeX table string
        """
        latex_output = []
        
        # Main results table
        latex_output.append(r"\begin{table}[htbp]")
        latex_output.append(r"\centering")
        latex_output.append(r"\caption{Impact of Degree of Freedom on Linguistic Metrics}")
        latex_output.append(r"\label{tab:dof_metrics}")
        latex_output.append(r"\begin{tabular}{lcccccc}")
        latex_output.append(r"\toprule")
        latex_output.append(r"DoF & MTLD & HD-D & Entropy & Sent. Len. & Lex. Dens. & FRE \\")
        latex_output.append(r"\midrule")
        
        for dof in sorted(aggregated_df['dof_value'].unique()):
            dof_data = aggregated_df[aggregated_df['dof_value'] == dof]
            
            if len(dof_data) > 0:
                row_data = dof_data.iloc[0]
                latex_output.append(
                    f"{dof:.1f} & "
                    f"{row_data.get('mtld_mean', 0):.2f} $\\pm$ {row_data.get('mtld_std', 0):.2f} & "
                    f"{row_data.get('hdd_mean', 0):.2f} $\\pm$ {row_data.get('hdd_std', 0):.2f} & "
                    f"{row_data.get('shannon_entropy_mean', 0):.2f} $\\pm$ {row_data.get('shannon_entropy_std', 0):.2f} & "
                    f"{row_data.get('mean_sentence_length_mean', 0):.1f} $\\pm$ {row_data.get('mean_sentence_length_std', 0):.1f} & "
                    f"{row_data.get('lexical_density_mean', 0):.3f} $\\pm$ {row_data.get('lexical_density_std', 0):.3f} & "
                    f"{row_data.get('flesch_reading_ease_mean', 0):.1f} $\\pm$ {row_data.get('flesch_reading_ease_std', 0):.1f} \\\\"
                )
        
        latex_output.append(r"\bottomrule")
        latex_output.append(r"\end{tabular}")
        latex_output.append(r"\end{table}")
        
        # Correlation table
        latex_output.append("")
        latex_output.append(r"\begin{table}[htbp]")
        latex_output.append(r"\centering")
        latex_output.append(r"\caption{Correlation Analysis: DoF vs. Linguistic Metrics}")
        latex_output.append(r"\label{tab:correlations}")
        latex_output.append(r"\begin{tabular}{lccc}")
        latex_output.append(r"\toprule")
        latex_output.append(r"Metric & Pearson's $r$ & Spearman's $\rho$ & Kendall's $\tau$ \\")
        latex_output.append(r"\midrule")
        
        for _, row in correlation_df.iterrows():
            significance_markers = ""
            if row['pearson_p'] < 0.001:
                significance_markers = "***"
            elif row['pearson_p'] < 0.01:
                significance_markers = "**"
            elif row['pearson_p'] < 0.05:
                significance_markers = "*"
            
            latex_output.append(
                f"{row['metric'].replace('_', ' ').title()} & "
                f"{row['pearson_r']:.3f}{significance_markers} & "
                f"{row['spearman_rho']:.3f} & "
                f"{row['kendall_tau']:.3f} \\\\"
            )
        
        latex_output.append(r"\bottomrule")
        latex_output.append(r"\multicolumn{4}{l}{\footnotesize * $p < 0.05$, ** $p < 0.01$, *** $p < 0.001$} \\")
        latex_output.append(r"\end{tabular}")
        latex_output.append(r"\end{table}")
        
        latex_string = "\n".join(latex_output)
        
        # Save to file
        latex_file = os.path.join(self.output_dir, 'tables.tex')
        with open(latex_file, 'w') as f:
            f.write(latex_string)
        
        logger.info(f"LaTeX tables saved to: {latex_file}")
        
        return latex_string
    
    def generate_summary_report(self, aggregated_df: pd.DataFrame,
                              correlation_df: pd.DataFrame,
                              anova_df: pd.DataFrame,
                              regression_results: Dict) -> str:
        """
        Generate comprehensive text summary report.
        
        Args:
            aggregated_df: Aggregated metrics
            correlation_df: Correlation analysis
            anova_df: ANOVA results
            regression_results: Regression analysis
            
        Returns:
            Summary report string
        """
        report = []
        report.append("=" * 80)
        report.append("DEGREE OF FREEDOM (DoF) IMPACT ANALYSIS - COMPREHENSIVE REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Dataset summary
        report.append("DATASET SUMMARY")
        report.append("-" * 40)
        total_samples = aggregated_df['n_samples'].sum()
        report.append(f"Total samples analyzed: {total_samples:,}")
        report.append(f"Number of DoF values: {len(aggregated_df['dof_value'].unique())}")
        report.append(f"DoF range: {aggregated_df['dof_value'].min():.2f} - {aggregated_df['dof_value'].max():.2f}")
        report.append(f"Models analyzed: {', '.join(aggregated_df['model_name'].unique())}")
        report.append("")
        
        # Key findings
        report.append("KEY FINDINGS")
        report.append("-" * 40)
        
        # Find strongest correlations
        strong_correlations = correlation_df[abs(correlation_df['pearson_r']) > 0.5]
        if not strong_correlations.empty:
            report.append("Strong correlations (|r| > 0.5):")
            for _, row in strong_correlations.iterrows():
                direction = "positive" if row['pearson_r'] > 0 else "negative"
                report.append(f"  - {row['metric']}: {direction} correlation (r = {row['pearson_r']:.3f}, p = {row['pearson_p']:.4f})")
        else:
            report.append("No strong correlations detected (|r| > 0.5)")
        
        report.append("")
        
        # Significant ANOVA results
        if not anova_df.empty:
            significant_anova = anova_df[anova_df['p_value'] < 0.05]
            if not significant_anova.empty:
                report.append("Statistically significant differences across DoF values:")
                for _, row in significant_anova.iterrows():
                    report.append(f"  - {row['metric']}: F = {row['f_statistic']:.3f}, p = {row['p_value']:.4f}, η² = {row['eta_squared']:.3f}")
            else:
                report.append("No statistically significant differences detected")
        
        report.append("")
        
        # Metric-specific analysis
        report.append("METRIC-SPECIFIC ANALYSIS")
        report.append("-" * 40)
        
        metrics = ['mtld', 'hdd', 'shannon_entropy', 'mean_sentence_length', 
                  'lexical_density', 'flesch_reading_ease']
        
        for metric in metrics:
            if f'{metric}_mean' not in aggregated_df.columns:
                continue
            
            report.append(f"\n{metric.upper().replace('_', ' ')}:")
            
            # Calculate change from min to max DoF
            min_dof_data = aggregated_df[aggregated_df['dof_value'] == aggregated_df['dof_value'].min()]
            max_dof_data = aggregated_df[aggregated_df['dof_value'] == aggregated_df['dof_value'].max()]
            
            if not min_dof_data.empty and not max_dof_data.empty:
                min_val = min_dof_data[f'{metric}_mean'].iloc[0]
                max_val = max_dof_data[f'{metric}_mean'].iloc[0]
                change = ((max_val - min_val) / min_val * 100) if min_val != 0 else 0
                
                report.append(f"  Range: {min_val:.3f} (DoF=0) → {max_val:.3f} (DoF=1)")
                report.append(f"  Change: {change:+.1f}%")
                
                # Add correlation info
                metric_corr = correlation_df[correlation_df['metric'] == metric]
                if not metric_corr.empty:
                    r = metric_corr['pearson_r'].iloc[0]
                    p = metric_corr['pearson_p'].iloc[0]
                    report.append(f"  Correlation with DoF: r = {r:.3f} (p = {p:.4f})")
                
                # Add regression info
                if metric in regression_results:
                    best_degree = regression_results[metric]['best_degree']
                    r2 = regression_results[metric]['models'][best_degree]['r2']
                    report.append(f"  Best fit: Polynomial degree {best_degree} (R² = {r2:.3f})")
        
        report.append("")
        report.append("=" * 80)
        report.append("END OF REPORT")
        report.append("=" * 80)
        
        report_string = "\n".join(report)
        
        # Save to file
        report_file = os.path.join(self.output_dir, 'summary_report.txt')
        with open(report_file, 'w') as f:
            f.write(report_string)
        
        logger.info(f"Summary report saved to: {report_file}")
        
        return report_string


def main():
    """
    Main execution function with command-line interface.
    """
    parser = argparse.ArgumentParser(
        description='Analyze Degree of Freedom impact on LLM text generation'
    )
    parser.add_argument(
        '--folder', 
        type=str, 
        default=r"C:\Users\dsng3\Documents\GitHub\DoF_Experiment\results\gemini\gemini-2.0-flash",
        help='Path to folder containing JSON files'
    )
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='dof_analysis_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--sample-limit', 
        type=int, 
        default=None,
        help='Limit number of samples per file (for testing)'
    )
    parser.add_argument(
        '--confidence', 
        type=float, 
        default=0.95,
        help='Confidence level for statistical tests'
    )
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize components
    logger.info("Initializing analysis framework...")
    calculator = AdvancedTextMetricsCalculator(confidence_level=args.confidence)
    processor = DataProcessor(calculator)
    analyzer = StatisticalAnalyzer()
    visualizer = AdvancedVisualization()
    reporter = ReportGenerator(args.output_dir)
    
    # Load data
    logger.info(f"Loading JSON files from: {args.folder}")
    all_data = processor.load_json_files(args.folder)
    
    if not all_data:
        logger.error("No valid JSON files found. Exiting.")
        return
    
    # Process data
    logger.info("Processing data and calculating metrics...")
    df = processor.process_data(all_data, sample_limit=args.sample_limit)
    
    if df.empty:
        logger.error("No valid samples processed. Exiting.")
        return
    
    # Outlier detection
    logger.info("Performing outlier detection...")
    df_with_outliers = processor.perform_outlier_detection(df)
    
    # Remove outliers for main analysis
    outlier_columns = [col for col in df_with_outliers.columns if '_outlier' in col]
    df_clean = df_with_outliers[~df_with_outliers[outlier_columns].any(axis=1)]
    logger.info(f"Removed {len(df) - len(df_clean)} outlier samples")
    
    # Aggregate metrics
    logger.info("Aggregating metrics...")
    aggregated_df = processor.aggregate_metrics(df_clean, confidence_level=args.confidence)
    
    # Statistical analyses
    logger.info("Performing statistical analyses...")
    
    # Correlation analysis
    correlation_df = analyzer.perform_correlation_analysis(aggregated_df)
    
    # ANOVA
    anova_df = analyzer.perform_anova(df_clean)
    
    # Regression analysis
    regression_results = analyzer.perform_regression_analysis(aggregated_df)
    
    # Generate visualizations
    logger.info("Generating visualizations...")
    
    # Main comprehensive figure
    visualizer.create_comprehensive_figure(
        aggregated_df, 
        correlation_df, 
        regression_results,
        os.path.join(args.output_dir, 'main_analysis.pdf')
    )
    
    # Correlation heatmap
    visualizer.create_correlation_heatmap(
        correlation_df,
        os.path.join(args.output_dir, 'correlations.pdf')
    )
    
    # Distribution plots
    visualizer.create_distribution_plots(
        df_clean,
        os.path.join(args.output_dir, 'distributions.pdf')
    )
    
    # Generate reports
    logger.info("Generating reports...")
    
    # LaTeX tables
    reporter.generate_latex_table(aggregated_df, correlation_df)
    
    # Text summary
    summary = reporter.generate_summary_report(
        aggregated_df, correlation_df, anova_df, regression_results
    )
    print("\n" + summary)
    
    # Save processed data
    logger.info("Saving processed data...")
    
    # Raw data
    df_clean.to_csv(
        os.path.join(args.output_dir, 'processed_data.csv'), 
        index=False
    )
    
    # Aggregated data
    aggregated_df.to_csv(
        os.path.join(args.output_dir, 'aggregated_metrics.csv'), 
        index=False
    )
    
    # Statistical results
    correlation_df.to_csv(
        os.path.join(args.output_dir, 'correlation_analysis.csv'), 
        index=False
    )
    anova_df.to_csv(
        os.path.join(args.output_dir, 'anova_results.csv'), 
        index=False
    )
    
    # Regression results
    with open(os.path.join(args.output_dir, 'regression_results.json'), 'w') as f:
        json.dump(regression_results, f, indent=2)
    
    logger.info(f"All results saved to: {args.output_dir}")
    logger.info("Analysis complete!")


if __name__ == "__main__":
    main()