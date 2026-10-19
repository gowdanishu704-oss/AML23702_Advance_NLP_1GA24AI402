"""
Domain Dataset: Financial & Business Intelligence Corpus for Word Sense Disambiguation
Curated polysemous terms and labeled sentences representing distinct domain senses vs. general senses.
"""

from typing import List, Dict, Any
import pandas as pd

TARGET_WORDS = ["bank", "interest", "stock", "capital", "bond", "yield", "plant", "balance", "credit"]

# Domain senses inventory mapping key words to their primary WordNet synset IDs
DOMAIN_SENSE_MAP = {
    "bank": {
        "bank.n.01": {
            "name": "Financial Institution",
            "gloss": "a financial institution that accepts deposits and channels the money into lending activities",
            "domain": "Finance / Banking",
            "keywords": ["deposit", "account", "loan", "money", "credit", "cash", "central", "reserve", "atm", "interest"]
        },
        "bank.n.02": {
            "name": "River Bank / Slope",
            "gloss": "sloping land (especially the slope beside a body of water)",
            "domain": "Geography / Nature",
            "keywords": ["river", "water", "stream", "mud", "shore", "flood", "grass", "fishing", "edge", "steep"]
        },
        "bank.n.07": {
            "name": "Arranged Flight / Tilt",
            "gloss": "a flight maneuver consisting of a steep turn to left or right",
            "domain": "Aviation",
            "keywords": ["plane", "pilot", "aircraft", "turn", "steep", "altitude", "flight"]
        }
    },
    "interest": {
        "interest.n.03": {
            "name": "Financial Charge / Yield",
            "gloss": "a fixed charge for borrowing money; usually a percentage of the amount borrowed",
            "domain": "Finance / Macroeconomics",
            "keywords": ["rate", "loan", "federal", "reserve", "borrow", "percentage", "basis", "points", "inflation", "debt"]
        },
        "interest.n.01": {
            "name": "Curiosity / Attention",
            "gloss": "a sense of concern with and curiosity about someone or something",
            "domain": "General / Psychology",
            "keywords": ["keen", "expressed", "piqued", "curiosity", "hobby", "attention", "passion", "fascinated"]
        },
        "interest.n.05": {
            "name": "Stake / Equity Share",
            "gloss": "(usually plural) a right or legal share of something; a financial involvement with something",
            "domain": "Corporate Governance",
            "keywords": ["controlling", "majority", "stake", "shares", "partner", "ownership", "venture", "holding"]
        }
    },
    "stock": {
        "stock.n.01": {
            "name": "Capital Stock / Equities",
            "gloss": "the capital raised by a corporation through the issue, distribution, and subscription of shares",
            "domain": "Finance / Equities",
            "keywords": ["shares", "market", "nasdaq", "rally", "dividend", "investor", "trading", "exchange", "index", "wall street"]
        },
        "stock.n.04": {
            "name": "Inventory / Supply",
            "gloss": "a supply of something available for future use",
            "domain": "Supply Chain / Retail",
            "keywords": ["warehouse", "inventory", "depleted", "goods", "shelves", "reorder", "storage", "replenish"]
        },
        "stock.n.05": {
            "name": "Broth / Culinary",
            "gloss": "liquid in which meat and vegetables are simmered; used as a basis for soups or sauces",
            "domain": "Culinary / Food",
            "keywords": ["soup", "simmer", "chicken", "beef", "pot", "cook", "vegetables", "broth", "flavor"]
        }
    },
    "capital": {
        "capital.n.01": {
            "name": "Financial Wealth / Assets",
            "gloss": "wealth in the form of money or other assets owned by a person or organization or contributed for a purpose such as starting a company or investing",
            "domain": "Finance / Venture",
            "keywords": ["venture", "investment", "raise", "funds", "liquidity", "equity", "seed", "assets", "working", "allocation"]
        },
        "capital.n.03": {
            "name": "Government Seat / Metropolis",
            "gloss": "a seat of government",
            "domain": "Geography / Politics",
            "keywords": ["city", "government", "parliament", "state", "country", "province", "travel", "seat"]
        },
        "capital.n.02": {
            "name": "Uppercase Letter",
            "gloss": "one of the large alphabetic characters used as the first letter in writing or printing",
            "domain": "Linguistics / Typography",
            "keywords": ["letter", "uppercase", "text", "font", "alphabet", "sentence", "grammar"]
        }
    },
    "bond": {
        "bond.n.01": {
            "name": "Debt Certificate / Fixed Income",
            "gloss": "a certificate of debt issued by a government or corporation in order to raise money; the issuer is required to pay a fixed interest at specified intervals",
            "domain": "Finance / Fixed Income",
            "keywords": ["treasury", "yield", "coupon", "maturity", "debt", "issuer", "sovereign", "credit", "investor"]
        },
        "bond.n.02": {
            "name": "Emotional Connection",
            "gloss": "an electrical force linking atoms or an emotional, social, or familial tie",
            "domain": "General / Psychology",
            "keywords": ["family", "friendship", "emotional", "trust", "mother", "child", "close", "affection"]
        },
        "chemical_bond.n.01": {
            "name": "Chemical Linkage",
            "gloss": "an electrical force linking atoms",
            "domain": "Chemistry",
            "keywords": ["molecule", "atom", "covalent", "ionic", "valence", "electron", "reaction"]
        }
    },
    "yield": {
        "yield.n.01": {
            "name": "Financial Return / Dividend",
            "gloss": "an amount of a product or merchandise resulting from a process; or financial return on an investment",
            "domain": "Finance / Earnings",
            "keywords": ["dividend", "percent", "annual", "return", "bonds", "treasury", "inversion", "curve", "earnings"]
        },
        "yield.v.01": {
            "name": "Give Way / Surrender",
            "gloss": "give or supply, or give way to arguments or demands",
            "domain": "General / Traffic",
            "keywords": ["driver", "intersection", "stop", "traffic", "pressure", "surrender", "pedestrian"]
        }
    },
    "plant": {
        "plant.n.01": {
            "name": "Flora / Botanical Organism",
            "gloss": "a living organism of the kind exemplified by trees, shrubs, herbs, grasses, ferns, and mosses",
            "domain": "Biology / Nature",
            "keywords": ["leaf", "soil", "water", "flower", "photosynthesis", "sunlight", "gardening", "botany", "green"]
        },
        "plant.n.02": {
            "name": "Factory / Manufacturing Facility",
            "gloss": "buildings for carrying on industrial labor",
            "domain": "Industrial / Corporate",
            "keywords": ["factory", "assembly", "manufacture", "automotive", "production", "workers", "shutdown", "machinery", "industrial"]
        }
    },
    "balance": {
        "balance.n.05": {
            "name": "Account Remainder / Financial Balance",
            "gloss": "equality between the debit and credit sides of an account, or the remaining sum of money",
            "domain": "Finance / Accounting",
            "keywords": ["account", "sheet", "bank", "credit", "ledger", "statement", "debit", "funds", "checking"]
        },
        "balance.n.01": {
            "name": "Physical Equilibrium",
            "gloss": "a state of equilibrium or equipoise; equal distribution of weight",
            "domain": "Physics / Fitness",
            "keywords": ["equilibrium", "gymnast", "fall", "gravity", "stable", "posture", "feet", "trip"]
        }
    },
    "credit": {
        "credit.n.02": {
            "name": "Loan / Financial Credit",
            "gloss": "an arrangement to receive cash, goods, or services now and pay for them in the future",
            "domain": "Finance / Lending",
            "keywords": ["card", "score", "loan", "debt", "rating", "limit", "bureau", "interest", "line"]
        },
        "credit.n.01": {
            "name": "Acknowledgment / Praise",
            "gloss": "approval or praise given to someone for some work or quality",
            "domain": "General / Social",
            "keywords": ["deserve", "given", "team", "achievement", "contribution", "authorship", "praise"]
        }
    }
}

# Annotated Corpus of Sentences for Evaluation and Supervised WSD
ANNOTATED_SENTENCES: List[Dict[str, Any]] = [
    # BANK - Financial (bank.n.01)
    {"id": 1, "word": "bank", "gold_synset": "bank.n.01", "sense_label": "financial_institution",
     "sentence": "The central bank raised interest rates yesterday to curb rising national inflation.",
     "domain": "finance"},
    {"id": 2, "word": "bank", "gold_synset": "bank.n.01", "sense_label": "financial_institution",
     "sentence": "Customers lined up outside the commercial bank to withdraw their cash deposits.",
     "domain": "finance"},
    {"id": 3, "word": "bank", "gold_synset": "bank.n.01", "sense_label": "financial_institution",
     "sentence": "She opened a high-yield savings account with an online investment bank.",
     "domain": "finance"},
    {"id": 4, "word": "bank", "gold_synset": "bank.n.01", "sense_label": "financial_institution",
     "sentence": "The investment bank underwrote the initial public offering for the fintech startup.",
     "domain": "finance"},
    {"id": 5, "word": "bank", "gold_synset": "bank.n.01", "sense_label": "financial_institution",
     "sentence": "Federal regulators shut down the insolvent regional bank over severe liquidity shortages.",
     "domain": "finance"},
    {"id": 6, "word": "bank", "gold_synset": "bank.n.01", "sense_label": "financial_institution",
     "sentence": "You can transfer money from your checking account using the mobile bank application.",
     "domain": "finance"},
    {"id": 7, "word": "bank", "gold_synset": "bank.n.01", "sense_label": "financial_institution",
     "sentence": "The bank approved our mortgage application after auditing our corporate credit score.",
     "domain": "finance"},
    {"id": 8, "word": "bank", "gold_synset": "bank.n.01", "sense_label": "financial_institution",
     "sentence": "European central bank officials warned of turbulent bond market conditions ahead.",
     "domain": "finance"},

    # BANK - River/Slope (bank.n.02)
    {"id": 9, "word": "bank", "gold_synset": "bank.n.02", "sense_label": "river_bank",
     "sentence": "We pitched our camping tents along the muddy bank of the swollen river.",
     "domain": "general"},
    {"id": 10, "word": "bank", "gold_synset": "bank.n.02", "sense_label": "river_bank",
     "sentence": "Heavy seasonal monsoon rains caused the river to overflow its steep grassy bank.",
     "domain": "general"},
    {"id": 11, "word": "bank", "gold_synset": "bank.n.02", "sense_label": "river_bank",
     "sentence": "The young fisherman sat quietly on the grassy canal bank waiting for trout to bite.",
     "domain": "general"},
    {"id": 12, "word": "bank", "gold_synset": "bank.n.02", "sense_label": "river_bank",
     "sentence": "Wild blooming flowers grew densely along the sandy bank of the winding stream.",
     "domain": "general"},
    {"id": 13, "word": "bank", "gold_synset": "bank.n.02", "sense_label": "river_bank",
     "sentence": "The canoe drifted smoothly toward the south river bank near the ancient willow trees.",
     "domain": "general"},
    {"id": 14, "word": "bank", "gold_synset": "bank.n.02", "sense_label": "river_bank",
     "sentence": "Children were skipping smooth stones across the water from the rocky river bank.",
     "domain": "general"},

    # INTEREST - Financial Charge (interest.n.03)
    {"id": 15, "word": "interest", "gold_synset": "interest.n.03", "sense_label": "financial_charge",
     "sentence": "The federal reserve increased benchmark interest rates by twenty-five basis points.",
     "domain": "finance"},
    {"id": 16, "word": "interest", "gold_synset": "interest.n.03", "sense_label": "financial_charge",
     "sentence": "High credit card interest charges can rapidly compound household debt burdens.",
     "domain": "finance"},
    {"id": 17, "word": "interest", "gold_synset": "interest.n.03", "sense_label": "financial_charge",
     "sentence": "The commercial mortgage loan carries a fixed annual interest rate of six percent.",
     "domain": "finance"},
    {"id": 18, "word": "interest", "gold_synset": "interest.n.03", "sense_label": "financial_charge",
     "sentence": "Treasury bond investors closely monitor yield fluctuations and real interest returns.",
     "domain": "finance"},
    {"id": 19, "word": "interest", "gold_synset": "interest.n.03", "sense_label": "financial_charge",
     "sentence": "Borrowers struggled to service the variable interest payments as policy tightened.",
     "domain": "finance"},
    {"id": 20, "word": "interest", "gold_synset": "interest.n.03", "sense_label": "financial_charge",
     "sentence": "The savings certificate accrued compound interest semi-annually in the credit union account.",
     "domain": "finance"},

    # INTEREST - Curiosity/Attention (interest.n.01)
    {"id": 21, "word": "interest", "gold_synset": "interest.n.01", "sense_label": "curiosity_attention",
     "sentence": "The new documentary film sparked widespread public interest in deep space exploration.",
     "domain": "general"},
    {"id": 22, "word": "interest", "gold_synset": "interest.n.01", "sense_label": "curiosity_attention",
     "sentence": "She has maintained an enthusiastic lifelong interest in classical Renaissance painting.",
     "domain": "general"},
    {"id": 23, "word": "interest", "gold_synset": "interest.n.01", "sense_label": "curiosity_attention",
     "sentence": "His curious hobby failed to generate any genuine interest among his peer group.",
     "domain": "general"},
    {"id": 24, "word": "interest", "gold_synset": "interest.n.01", "sense_label": "curiosity_attention",
     "sentence": "The professor listened with keen interest as the graduate student presented novel hypotheses.",
     "domain": "general"},
    {"id": 25, "word": "interest", "gold_synset": "interest.n.01", "sense_label": "curiosity_attention",
     "sentence": "Visitors showed immense interest in the historical artifacts excavated at the ancient ruin.",
     "domain": "general"},

    # STOCK - Equities (stock.n.01)
    {"id": 26, "word": "stock", "gold_synset": "stock.n.01", "sense_label": "capital_stock_shares",
     "sentence": "Tech company stock prices surged on Wall Street following blowout quarterly earnings.",
     "domain": "finance"},
    {"id": 27, "word": "stock", "gold_synset": "stock.n.01", "sense_label": "capital_stock_shares",
     "sentence": "Retail investors purchased additional common stock during the market pullback.",
     "domain": "finance"},
    {"id": 28, "word": "stock", "gold_synset": "stock.n.01", "sense_label": "capital_stock_shares",
     "sentence": "The corporation announced a generous stock repurchase program to boost shareholder value.",
     "domain": "finance"},
    {"id": 29, "word": "stock", "gold_synset": "stock.n.01", "sense_label": "capital_stock_shares",
     "sentence": "Trading volume on the Tokyo stock exchange reached unprecedented record highs today.",
     "domain": "finance"},
    {"id": 30, "word": "stock", "gold_synset": "stock.n.01", "sense_label": "capital_stock_shares",
     "sentence": "Diversifying your investment portfolio between stock indexes and treasury bonds mitigates risk.",
     "domain": "finance"},

    # STOCK - Inventory/Supply (stock.n.04)
    {"id": 31, "word": "stock", "gold_synset": "stock.n.04", "sense_label": "inventory_supply",
     "sentence": "The retail warehouse is completely out of stock due to severe global shipping delays.",
     "domain": "general"},
    {"id": 32, "word": "stock", "gold_synset": "stock.n.04", "sense_label": "inventory_supply",
     "sentence": "Store clerks worked overtime throughout the weekend to restock empty supermarket shelves.",
     "domain": "general"},
    {"id": 33, "word": "stock", "gold_synset": "stock.n.04", "sense_label": "inventory_supply",
     "sentence": "We must replenish our depleted stock of medical emergency supplies immediately.",
     "domain": "general"},
    {"id": 34, "word": "stock", "gold_synset": "stock.n.04", "sense_label": "inventory_supply",
     "sentence": "Automated inventory management algorithms keep a safety buffer stock in regional distribution hubs.",
     "domain": "general"},

    # STOCK - Culinary Broth (stock.n.05)
    {"id": 35, "word": "stock", "gold_synset": "stock.n.05", "sense_label": "culinary_broth",
     "sentence": "The chef simmered roasted chicken bones and fresh aromatic vegetables to create a flavorful soup stock.",
     "domain": "general"},
    {"id": 36, "word": "stock", "gold_synset": "stock.n.05", "sense_label": "culinary_broth",
     "sentence": "Add four cups of savory beef stock to the pot before bringing the stew to a boil.",
     "domain": "general"},

    # CAPITAL - Financial Wealth/Assets (capital.n.01)
    {"id": 37, "word": "capital", "gold_synset": "capital.n.01", "sense_label": "financial_capital",
     "sentence": "The venture capital firm deployed fifty million dollars into promising artificial intelligence startups.",
     "domain": "finance"},
    {"id": 38, "word": "capital", "gold_synset": "capital.n.01", "sense_label": "financial_capital",
     "sentence": "Commercial banks must maintain adequate tier-one capital reserves to absorb sudden financial losses.",
     "domain": "finance"},
    {"id": 39, "word": "capital", "gold_synset": "capital.n.01", "sense_label": "financial_capital",
     "sentence": "Raising external equity capital enabled the enterprise to expand manufacturing capacity globally.",
     "domain": "finance"},
    {"id": 40, "word": "capital", "gold_synset": "capital.n.01", "sense_label": "financial_capital",
     "sentence": "Capital expenditures for semiconductor equipment exceeded quarterly guidance by twelve percent.",
     "domain": "finance"},
    {"id": 41, "word": "capital", "gold_synset": "capital.n.01", "sense_label": "financial_capital",
     "sentence": "Efficient working capital management reduced the operating cash cycle for corporate treasurers.",
     "domain": "finance"},

    # CAPITAL - Government City/Seat (capital.n.03)
    {"id": 42, "word": "capital", "gold_synset": "capital.n.03", "sense_label": "government_city",
     "sentence": "Government delegates convened in the national capital to sign the multilateral peace accord.",
     "domain": "general"},
    {"id": 43, "word": "capital", "gold_synset": "capital.n.03", "sense_label": "government_city",
     "sentence": "Paris is widely renowned as the historic cultural and fashion capital of Western Europe.",
     "domain": "general"},
    {"id": 44, "word": "capital", "gold_synset": "capital.n.03", "sense_label": "government_city",
     "sentence": "Thousands of enthusiastic tourists visited the historic parliament buildings in the state capital.",
     "domain": "general"},
    {"id": 45, "word": "capital", "gold_synset": "capital.n.03", "sense_label": "government_city",
     "sentence": "The relocation of administrative ministries to the new administrative capital finished this year.",
     "domain": "general"},

    # BOND - Debt Certificate (bond.n.01)
    {"id": 46, "word": "bond", "gold_synset": "bond.n.01", "sense_label": "debt_bond",
     "sentence": "Institutional investors bought ten-year sovereign treasury bond securities as yields spiked.",
     "domain": "finance"},
    {"id": 47, "word": "bond", "gold_synset": "bond.n.01", "sense_label": "debt_bond",
     "sentence": "The municipal city government issued a green infrastructure bond to fund clean transit projects.",
     "domain": "finance"},
    {"id": 48, "word": "bond", "gold_synset": "bond.n.01", "sense_label": "debt_bond",
     "sentence": "Corporate high-yield junk bond spreads widened amid growing concerns over corporate debt defaults.",
     "domain": "finance"},
    {"id": 49, "word": "bond", "gold_synset": "bond.n.01", "sense_label": "debt_bond",
     "sentence": "The bond coupon rate pays fixed interest semi-annually until final maturity in twenty years.",
     "domain": "finance"},

    # BOND - Emotional Tie (bond.n.02)
    {"id": 50, "word": "bond", "gold_synset": "bond.n.02", "sense_label": "emotional_tie",
     "sentence": "A shared childhood experience forged an unbreakable lifelong friendship bond between the brothers.",
     "domain": "general"},
    {"id": 51, "word": "bond", "gold_synset": "bond.n.02", "sense_label": "emotional_tie",
     "sentence": "The emotional bond between a newborn baby and its nurturing mother develops within hours.",
     "domain": "general"},
    {"id": 52, "word": "bond", "gold_synset": "bond.n.02", "sense_label": "emotional_tie",
     "sentence": "Adversity and shared hardship created a deep sense of psychological bond among the crew members.",
     "domain": "general"},

    # PLANT - Factory / Facility (plant.n.02)
    {"id": 53, "word": "plant", "gold_synset": "plant.n.02", "sense_label": "manufacturing_facility",
     "sentence": "The automotive manufacturer opened an electric vehicle assembly plant in Ohio.",
     "domain": "finance"},
    {"id": 54, "word": "plant", "gold_synset": "plant.n.02", "sense_label": "manufacturing_facility",
     "sentence": "Industrial output declined after severe equipment maintenance forced a chemical plant shutdown.",
     "domain": "finance"},
    {"id": 55, "word": "plant", "gold_synset": "plant.n.02", "sense_label": "manufacturing_facility",
     "sentence": "The semiconductor fabrication plant requires billions of dollars in advanced lithography tools.",
     "domain": "finance"},
    {"id": 56, "word": "plant", "gold_synset": "plant.n.02", "sense_label": "manufacturing_facility",
     "sentence": "Union workers at the industrial steel processing plant voted to ratify the labor contract.",
     "domain": "finance"},

    # PLANT - Botanical Organism (plant.n.01)
    {"id": 57, "word": "plant", "gold_synset": "plant.n.01", "sense_label": "botanical_flora",
     "sentence": "Ensure the indoor potted plant receives sufficient sunlight and moist organic soil.",
     "domain": "general"},
    {"id": 58, "word": "plant", "gold_synset": "plant.n.01", "sense_label": "botanical_flora",
     "sentence": "Botanists discovered a rare carnivorous flowering plant in the tropical rainforest canopy.",
     "domain": "general"},
    {"id": 59, "word": "plant", "gold_synset": "plant.n.01", "sense_label": "botanical_flora",
     "sentence": "Water the delicate garden plant regularly to encourage healthy root growth and green foliage.",
     "domain": "general"},

    # BALANCE - Accounting Balance (balance.n.05)
    {"id": 60, "word": "balance", "gold_synset": "balance.n.05", "sense_label": "account_balance",
     "sentence": "Please log into the banking portal to verify your outstanding credit card balance.",
     "domain": "finance"},
    {"id": 61, "word": "balance", "gold_synset": "balance.n.05", "sense_label": "account_balance",
     "sentence": "Auditors examined the corporate balance sheet to inspect current cash reserves and liabilities.",
     "domain": "finance"},
    {"id": 62, "word": "balance", "gold_synset": "balance.n.05", "sense_label": "account_balance",
     "sentence": "Her checking account balance dropped below the minimum required monthly threshold.",
     "domain": "finance"},

    # BALANCE - Physical Equilibrium (balance.n.01)
    {"id": 63, "word": "balance", "gold_synset": "balance.n.01", "sense_label": "physical_equilibrium",
     "sentence": "The gymnast slipped on the narrow beam and struggled desperately to maintain her physical balance.",
     "domain": "general"},
    {"id": 64, "word": "balance", "gold_synset": "balance.n.01", "sense_label": "physical_equilibrium",
     "sentence": "Riding a bicycle along bumpy gravel requires coordination and keen bodily balance.",
     "domain": "general"},

    # CREDIT - Loan/Financial Credit (credit.n.02)
    {"id": 65, "word": "credit", "gold_synset": "credit.n.02", "sense_label": "financial_credit",
     "sentence": "The commercial bank extended a revolving credit line to help the corporation meet payroll.",
     "domain": "finance"},
    {"id": 66, "word": "credit", "gold_synset": "credit.n.02", "sense_label": "financial_credit",
     "sentence": "Rating agencies downgraded sovereign debt ratings, triggering tighter credit conditions worldwide.",
     "domain": "finance"},
    {"id": 67, "word": "credit", "gold_synset": "credit.n.02", "sense_label": "financial_credit",
     "sentence": "Maintaining a high credit score enables borrowers to secure favorable interest rates on mortgages.",
     "domain": "finance"},

    # CREDIT - Praise / Acknowledgment (credit.n.01)
    {"id": 68, "word": "credit", "gold_synset": "credit.n.01", "sense_label": "praise_recognition",
     "sentence": "The lead researcher graciously gave credit to her graduate lab assistants for the scientific breakthrough.",
     "domain": "general"},
    {"id": 69, "word": "credit", "gold_synset": "credit.n.01", "sense_label": "praise_recognition",
     "sentence": "He took full personal credit for the collaborative marketing campaign despite doing minimal work.",
     "domain": "general"}
]

def get_annotated_dataframe() -> pd.DataFrame:
    """Returns the labeled dataset as a pandas DataFrame."""
    return pd.DataFrame(ANNOTATED_SENTENCES)

def get_sentences_for_word(word: str) -> List[Dict[str, Any]]:
    """Returns sample sentences for a given target word."""
    return [s for s in ANNOTATED_SENTENCES if s["word"].lower() == word.lower()]

def get_domain_senses(word: str) -> Dict[str, Any]:
    """Returns the curated domain sense mapping for a target word if available."""
    return DOMAIN_SENSE_MAP.get(word.lower(), {})
