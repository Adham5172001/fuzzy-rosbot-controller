"""
Fuzzy Rule Base for ROSbot Navigation
Author: Adham Aboulkheir | University of Essex
"""
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class FuzzyRule:
    """A single fuzzy IF-THEN rule."""
    antecedents: Dict[str, str]   # {"distance": "NEAR", "speed": "HIGH"}
    consequent: str               # "STOP"
    weight: float = 1.0
    description: str = ""
    
    def to_natural_language(self) -> str:
        conditions = [f"{var} is {term}" for var, term in self.antecedents.items()]
        return f"IF {' AND '.join(conditions)} THEN command is {self.consequent}"


# Complete rule base for the fuzzy navigation controller
NAVIGATION_RULES = [
    FuzzyRule(
        antecedents={"distance": "NEAR", "speed": "HIGH"},
        consequent="STOP",
        weight=1.0,
        description="Obstacle very close and moving fast — emergency stop"
    ),
    FuzzyRule(
        antecedents={"distance": "NEAR", "speed": "LOW"},
        consequent="SLOW",
        weight=1.0,
        description="Obstacle close but moving slowly — reduce speed further"
    ),
    FuzzyRule(
        antecedents={"distance": "MEDIUM", "speed": "HIGH"},
        consequent="SLOW",
        weight=1.0,
        description="Obstacle at medium distance, moving fast — slow down"
    ),
    FuzzyRule(
        antecedents={"distance": "MEDIUM", "speed": "LOW"},
        consequent="MODERATE",
        weight=1.0,
        description="Obstacle at medium distance, moving slowly — proceed moderately"
    ),
    FuzzyRule(
        antecedents={"distance": "FAR", "speed": "HIGH"},
        consequent="FAST",
        weight=1.0,
        description="No obstacle ahead, moving fast — maintain speed"
    ),
    FuzzyRule(
        antecedents={"distance": "FAR", "speed": "LOW"},
        consequent="FAST",
        weight=1.0,
        description="No obstacle ahead, moving slowly — accelerate"
    ),
]


def evaluate_rules(rules: List[FuzzyRule], inputs: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """
    Evaluate all rules given fuzzified input values.
    
    Parameters
    ----------
    rules  : list of FuzzyRule objects
    inputs : {"distance": {"NEAR": 0.8, "MEDIUM": 0.2, "FAR": 0.0}, 
              "speed":    {"LOW": 0.3, "HIGH": 0.7}}
    
    Returns
    -------
    output_activations : {"STOP": 0.7, "SLOW": 0.2, "MODERATE": 0.0, "FAST": 0.0}
    """
    activations = {}
    
    for rule in rules:
        # Compute rule activation (minimum of antecedent memberships)
        activation = rule.weight
        for var, term in rule.antecedents.items():
            activation = min(activation, inputs[var].get(term, 0.0))
        
        # Aggregate (maximum)
        consequent = rule.consequent
        activations[consequent] = max(activations.get(consequent, 0.0), activation)
    
    return activations


if __name__ == "__main__":
    print("Fuzzy Rule Base Demo")
    print("=" * 40)
    print(f"Total rules: {len(NAVIGATION_RULES)}")
    print("\nRule Base:")
    for i, rule in enumerate(NAVIGATION_RULES, 1):
        print(f"  Rule {i}: {rule.to_natural_language()}")
    
    # Example evaluation
    inputs = {
        "distance": {"NEAR": 0.8, "MEDIUM": 0.2, "FAR": 0.0},
        "speed":    {"LOW": 0.3, "HIGH": 0.7}
    }
    activations = evaluate_rules(NAVIGATION_RULES, inputs)
    print(f"\nExample (obstacle=30cm, speed=0.8):")
    print(f"  Activations: {activations}")
