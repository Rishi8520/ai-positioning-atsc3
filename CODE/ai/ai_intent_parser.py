# ===============================================================================
# FILE: ai_intent_parser.py
# MODULE: Intent Parser for PPaaS (Precise Positioning-as-a-Service)
# AUTHOR: Tarunika D (AI/ML Systems)
# DATE: January 2026
# PURPOSE: Parse operator/fleet intents to canonical positioning goals
# PRODUCTION: Phase 3 - Ready for Deployment
# V2 UPDATES: Config integration, enhanced validation, constraint checking
# ===============================================================================

import re
import logging
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
from enum import Enum

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

# V2 Configuration
try:
    from config_v2 import cfg
    USE_V2_CONFIG = True
except ImportError:
    USE_V2_CONFIG = False
    cfg = None

# ===============================================================================
# LOGGING CONFIGURATION
# ===============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===============================================================================
# ENUMS & CONSTANTS
# ===============================================================================

class IntentType(Enum):
    """Canonical positioning intents"""
    MAXIMIZE_ACCURACY = "maximize_accuracy"
    MAXIMIZE_RELIABILITY = "maximize_reliability"
    OPTIMIZE_SPECTRUM = "optimize_spectrum"


# ===============================================================================
# DATA CLASSES
# ===============================================================================

@dataclass
class IntentConstraints:
    """Extracted constraints from parsed intent (V2 validated)"""
    target_hpe_cm: float  # Horizontal positioning error target (cm)
    min_availability_pct: float  # Minimum RTK FIX availability (%)
    max_spectrum_mbps: float  # Maximum spectrum budget (Mbps)
    max_convergence_sec: float  # Maximum convergence time (seconds)
    preferred_region: Optional[str] = None  # Geographic preference
    is_valid: bool = False  # NEW: Validation flag
    validation_notes: str = ""  # NEW: Validation feedback
    
    def validate(self) -> bool:
        """
        NEW: Validate constraints are within reasonable ranges
        
        Returns:
            True if valid, updates is_valid flag
        """
        valid = True
        notes = []
        
        if self.target_hpe_cm < 0.1 or self.target_hpe_cm > 100.0:
            valid = False
            notes.append(f"HPE {self.target_hpe_cm}cm out of range [0.1, 100.0]")
        
        if self.min_availability_pct < 50.0 or self.min_availability_pct > 99.99:
            valid = False
            notes.append(f"Availability {self.min_availability_pct}% out of range [50, 99.99]")
        
        if self.max_spectrum_mbps < 0.1 or self.max_spectrum_mbps > 10.0:
            valid = False
            notes.append(f"Spectrum {self.max_spectrum_mbps}Mbps out of range [0.1, 10.0]")
        
        if self.max_convergence_sec < 1.0 or self.max_convergence_sec > 300.0:
            valid = False
            notes.append(f"Convergence {self.max_convergence_sec}s out of range [1, 300]")
        
        self.is_valid = valid
        self.validation_notes = "; ".join(notes) if notes else "Valid"
        
        return valid


@dataclass
class CanonicalIntent:
    """Structured output from intent parser (V2 enhanced)"""
    intent_type: IntentType
    confidence: float  # Confidence score (0-1)
    constraints: IntentConstraints
    raw_text: str
    intent_embedding: np.ndarray  # 32D embedding
    reasoning: str  # Explanation of parsing decision
    embedding_dim: int = 32  # NEW: Track embedding dimension
    model_version: str = "v2"  # NEW: Track model version


# ===============================================================================
# INTENT RECOGNITION PATTERNS
# ===============================================================================

INTENT_PATTERNS = {
    IntentType.MAXIMIZE_ACCURACY: {
        "keywords": [
            r"sub-?\d+\s*cm", r"centimetre", r"centimeter", r"accuracy",
            r"precise", r"drift", r"high precision", r"cm-level", r"mmHPE"
        ],
        "phrases": [
            r"drone\s*(inspection|survey|delivery)",
            r"autonomous\s*delivery",
            r"survey\s*(grade|accuracy)",
            r"sub-\d+\s*cm\s*accuracy",
            r"centimetre-level",
            r"RTK\s*(fix|FIX)"
        ]
    },
    IntentType.MAXIMIZE_RELIABILITY: {
        "keywords": [
            r"reliability", r"continuity", r"availability", r"robust",
            r"urban\s*canyon", r"tunnel", r"blockage", r"multipath",
            r"degradation", r"outage", r"fail.*safe"
        ],
        "phrases": [
            r"urban\s*canyon",
            r"tunnel.*corridor",
            r"dense.*urban",
            r"multipath\s*environment",
            r"signal\s*loss",
            r"minimize.*outage",
            r"continuity.*service"
        ]
    },
    IntentType.OPTIMIZE_SPECTRUM: {
        "keywords": [
            r"spectrum", r"bandwidth", r"efficient", r"minimize",
            r"reduce.*resource", r"low-?bandwidth", r"coverage",
            r"rural", r"efficiency"
        ],
        "phrases": [
            r"minimize\s*spectrum",
            r"bandwidth\s*efficient",
            r"rural\s*coverage",
            r"low-?bandwidth",
            r"spectrum\s*optimization",
            r"coverage\s*area"
        ]
    }
}

# ===============================================================================
# INTENT EMBEDDING MODEL
# ===============================================================================

class IntentEmbeddingModel(nn.Module):
    """Neural network for learning intent embeddings"""
    
    def __init__(self, input_dim=768, hidden_dim=256, output_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, x):
        return self.encoder(x)


# ===============================================================================
# INTENT PARSER CLASS
# ===============================================================================

class IntentParser:
    """Intent parser for PPaaS system (V2 compatible)"""
    
    def __init__(self, pretrained_model="sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize intent parser with pretrained transformer model
        
        Args:
            pretrained_model: HuggingFace model ID for embeddings
        """
        logger.info(f"Loading pretrained model: {pretrained_model}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.embedding_model = AutoModel.from_pretrained(pretrained_model)
        self.embedding_model.eval()
        
        # Intent embedding network
        self.intent_encoder = IntentEmbeddingModel(input_dim=384, output_dim=32)
        self.intent_encoder.eval()
        
        # V2: Use config if available
        if USE_V2_CONFIG and cfg:
            confidence_threshold = cfg.intent.confidence_threshold
            embedding_dim = cfg.intent.embedding_dim
        else:
            confidence_threshold = 0.6
            embedding_dim = 32
        
        self.confidence_threshold = confidence_threshold
        self.embedding_dim = embedding_dim
        
        # V2: Enhanced threshold mapping
        self.constraint_thresholds = {
            IntentType.MAXIMIZE_ACCURACY: {
                "target_hpe_cm": 3.0,
                "min_availability_pct": 95.0,
                "max_spectrum_mbps": 2.0,
                "max_convergence_sec": 30.0
            },
            IntentType.MAXIMIZE_RELIABILITY: {
                "target_hpe_cm": 10.0,
                "min_availability_pct": 98.0,
                "max_spectrum_mbps": 3.0,
                "max_convergence_sec": 45.0
            },
            IntentType.OPTIMIZE_SPECTRUM: {
                "target_hpe_cm": 15.0,
                "min_availability_pct": 90.0,
                "max_spectrum_mbps": 1.5,
                "max_convergence_sec": 60.0
            }
        }
        
        logger.info(f"IntentParser initialized (confidence threshold={confidence_threshold})")
    
    def _get_text_embedding(self, text: str) -> np.ndarray:
        """Get 384D embedding from pretrained transformer"""
        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                max_length=512,
                truncation=True,
                padding=True,
                return_tensors="pt"
            )
            outputs = self.embedding_model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
        
        return embeddings[0]
    
    def _extract_numeric_constraints(self, text: str) -> Dict[str, float]:
        """Extract numeric values from text"""
        constraints = {}
        
        # HPE/accuracy extraction
        hpe_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:cm|centimeter)', text, re.IGNORECASE)
        if hpe_match:
            constraints['target_hpe_cm'] = float(hpe_match.group(1))
        
        # Availability extraction
        avail_match = re.search(r'(\d+(?:\.\d+)?)\s*%\s*(?:availability|uptime|FIX)', text, re.IGNORECASE)
        if avail_match:
            constraints['min_availability_pct'] = float(avail_match.group(1))
        
        # Spectrum extraction
        spectrum_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:Mbps|MB/s)', text, re.IGNORECASE)
        if spectrum_match:
            constraints['max_spectrum_mbps'] = float(spectrum_match.group(1))
        
        # Convergence time extraction
        conv_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:second|sec|s)\s*(?:to FIX|convergence)', text, re.IGNORECASE)
        if conv_match:
            constraints['max_convergence_sec'] = float(conv_match.group(1))
        
        return constraints
    
    def _calculate_intent_confidence(self, text: str, intent_type: IntentType) -> float:
        """Calculate confidence score for detected intent"""
        patterns = INTENT_PATTERNS[intent_type]
        
        keyword_score = 0.0
        for keyword in patterns["keywords"]:
            if re.search(keyword, text, re.IGNORECASE):
                keyword_score += 0.1
        
        phrase_score = 0.0
        for phrase in patterns["phrases"]:
            if re.search(phrase, text, re.IGNORECASE):
                phrase_score += 0.2
        
        total_score = min(1.0, keyword_score + phrase_score)
        return total_score
    
    def _generate_intent_embedding(self, text: str, intent_type: IntentType) -> np.ndarray:
        """Generate 32D intent-aware embedding"""
        # Get base embedding (384D)
        base_embedding = self._get_text_embedding(text)
        
        # Convert to tensor and pass through encoder
        with torch.no_grad():
            embedding_tensor = torch.from_numpy(base_embedding).unsqueeze(0).float()
            intent_embedding = self.intent_encoder(embedding_tensor).numpy()[0]
        
        return intent_embedding
    
    def parse(self, intent_text: str) -> CanonicalIntent:
        """
        Parse free-text intent to structured format (V2 with validation)
        
        Args:
            intent_text: Operator's natural language intent
        
        Returns:
            CanonicalIntent with parsed intent type, validated constraints, embedding
        """
        logger.info(f"Parsing intent: {intent_text[:100]}...")
        
        # Detect intent type with confidence scores
        intent_scores = {}
        for intent_type in IntentType:
            confidence = self._calculate_intent_confidence(intent_text, intent_type)
            intent_scores[intent_type] = confidence
        
        # Select intent with highest confidence
        detected_intent = max(intent_scores, key=intent_scores.get)
        confidence = intent_scores[detected_intent]
        
        # Extract numeric constraints from text
        extracted_constraints = self._extract_numeric_constraints(intent_text)
        
        # Apply defaults from intent type thresholds
        thresholds = self.constraint_thresholds[detected_intent]
        constraints_dict = {
            "target_hpe_cm": extracted_constraints.get("target_hpe_cm", thresholds["target_hpe_cm"]),
            "min_availability_pct": extracted_constraints.get("min_availability_pct", thresholds["min_availability_pct"]),
            "max_spectrum_mbps": extracted_constraints.get("max_spectrum_mbps", thresholds["max_spectrum_mbps"]),
            "max_convergence_sec": extracted_constraints.get("max_convergence_sec", thresholds["max_convergence_sec"])
        }
        
        constraints = IntentConstraints(**constraints_dict)
        
        # NEW: Validate constraints
        is_valid = constraints.validate()
        if not is_valid:
            logger.warning(f"Constraint validation failed: {constraints.validation_notes}")
        
        # Generate intent embedding
        intent_embedding = self._generate_intent_embedding(intent_text, detected_intent)
        
        # Generate reasoning
        reasoning = (
            f"Detected intent: {detected_intent.value} (confidence: {confidence:.2f}) | "
            f"HPE target: {constraints.target_hpe_cm}cm | "
            f"Availability: {constraints.min_availability_pct}% | "
            f"Spectrum: {constraints.max_spectrum_mbps}Mbps | "
            f"Status: {'✓ Valid' if is_valid else '✗ ' + constraints.validation_notes}"
        )
        
        logger.info(reasoning)
        
        return CanonicalIntent(
            intent_type=detected_intent,
            confidence=confidence,
            constraints=constraints,
            raw_text=intent_text,
            intent_embedding=intent_embedding,
            reasoning=reasoning,
            embedding_dim=self.embedding_dim,
            model_version="v2"
        )
    
    def to_dict(self, canonical_intent: CanonicalIntent) -> Dict:
        """Serialize CanonicalIntent to dictionary (V2 enhanced)"""
        return {
            "intent_type": canonical_intent.intent_type.value,
            "confidence": canonical_intent.confidence,
            "constraints": {
                **asdict(canonical_intent.constraints),
                "is_valid": canonical_intent.constraints.is_valid,
                "validation_notes": canonical_intent.constraints.validation_notes
            },
            "raw_text": canonical_intent.raw_text,
            "intent_embedding": canonical_intent.intent_embedding.tolist(),
            "reasoning": canonical_intent.reasoning,
            "embedding_dim": canonical_intent.embedding_dim,
            "model_version": canonical_intent.model_version
        }


# ===============================================================================
# MAIN EXECUTION FOR TESTING
# ===============================================================================

if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("INTENT PARSER - STANDALONE TEST")
    logger.info("=" * 80)
    
    parser = IntentParser()
    
    # Test intents
    test_intents = [
        "I need sub-3cm accuracy for drone inspection at 95% availability with minimal spectrum",
        "Maximize reliability in urban canyons with tunnels. Signal loss is common here",
        "Minimize ATSC spectrum usage while maintaining 90% RTK FIX availability in rural areas"
    ]
    
    for test_intent in test_intents:
        logger.info(f"\nInput: {test_intent}")
        result = parser.parse(test_intent)
        logger.info(f"Output: {parser.to_dict(result)}")
    
    logger.info("\n" + "=" * 80)
    logger.info("TEST COMPLETE")
    logger.info("=" * 80)
