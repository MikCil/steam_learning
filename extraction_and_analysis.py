import json
import os
import spacy
import pandas as pd
import steamreviews  # Added library

# --- CONFIGURATION ---
# Pairs of Systems-Heavy Games (Co-op vs. Single Player)
GAME_CORPUS = {
    # Pair 1: Automation/Strategy
    'Factorio': {'id': 427520, 'type': 'Co-op'},
    'Dyson Sphere Program': {'id': 1366540, 'type': 'Single-Player'},
    
    # Pair 2: Survival/Systems
    'Dont Starve Together': {'id': 322330, 'type': 'Co-op'},
    'Subnautica': {'id': 264710, 'type': 'Single-Player'},
    
    # Pair 3: Puzzle/Logic
    'Portal 2': {'id': 620, 'type': 'Co-op'},
    'The Talos Principle': {'id': 257510, 'type': 'Single-Player'}
}

# Settings
REVIEW_LIMIT = 5000000  # Analysis cap per game to keep processing time reasonable
DATA_FOLDER = 'data'  # Default folder created by steamreviews library
NLP_MODEL = spacy.load("en_core_web_lg")

# --- LEXICONS ---
LEARNING_LEMMAS = {'learn', 'teach', 'figure', 'realize', 'discover', 'understand', 'grasp', 'master', 'solve'}
PERCEPTION_LEMMAS = {'see', 'saw', 'notice', 'hear', 'sound', 'look', 'watch', 'observe', 'spot', 'visualize', 'listen'}
EPISTEMIC_LEMMAS = {'know', 'think', 'conclude', 'guess', 'theory', 'assume', 'plan', 'analyze', 'logic'}

# Subject pronouns (Who is doing the learning?)
PLAYER_SUBJECTS = {'i', 'we', 'me', 'us', 'myself', 'ourselves'}

# Shared context indicators (To search in the whole clause)
SHARED_INDICATORS = {'we', 'us', 'our', 'ours', 'ourselves', 'friend', 'partner', 'teammate'}

def fetch_game_reviews(app_id, limit=REVIEW_LIMIT):
    """
    Fetches reviews from Steam API if local data is missing.
    """
    print(f"  [...] Downloading reviews for App ID {app_id}...")
    request_params = dict(
        language='english',
        filter='recent',      # specific filter for "most helpful"
        review_type='all', 
        purchase_type='all' 
    )
    
    # Download reviews
    review_dict, _ = steamreviews.download_reviews_for_app_id(app_id, chosen_request_params=request_params)
    
    reviews_list = []
    
    # Check if 'reviews' key exists in the dictionary
    if review_dict and 'reviews' in review_dict:
        current_reviews = review_dict['reviews']
        
        # NOTE: Sometimes the library nests by App ID. 
        if str(app_id) in current_reviews:
            current_reviews = current_reviews[str(app_id)]

        # Sort by helpfulness (votes_up)
        # We use .get() to be safe, though 'votes_up' is standard
        sorted_reviews = sorted(current_reviews.values(), key=lambda x: x.get('votes_up', 0), reverse=True)
        
        for r in sorted_reviews[:limit]:
            reviews_list.append(r.get('review', ''))
            
    return reviews_list

def get_game_data(app_id, limit=REVIEW_LIMIT):
    """
    Loads reviews from local JSON if available; otherwise downloads them.
    """
    filename = os.path.join(DATA_FOLDER, f"review_{app_id}.json")
    
    # 1. Try Local File
    if os.path.exists(filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if 'reviews' in data:
                print(f"  [+] Loaded local file: {filename}")
                reviews_map = data['reviews']
                reviews_list = list(reviews_map.values())
                reviews_list.sort(key=lambda x: x.get('votes_up', 0), reverse=True)
                return [r.get('review', '') for r in reviews_list[:limit]]
            else:
                print(f"  [!] 'reviews' key not found in {filename}. Attempting redownload...")
        except Exception as e:
            print(f"  [!] Error reading {filename}: {e}. Attempting redownload...")

    # 2. Fallback to Download
    print(f"  [!] Local data unavailable or corrupt for {app_id}. Fetching from Steam...")
    return fetch_game_reviews(app_id, limit)

def analyze_learning_claims(text):
    """
    Identifies learning claims, checks for shared context in the subtree,
    and highlights the verb in the full text.
    """
    doc = NLP_MODEL(text)
    findings = []
    
    for sent in doc.sents:
        learning_tokens = [t for t in sent if t.lemma_ in LEARNING_LEMMAS]
        
        if not learning_tokens:
            continue
            
        for token in learning_tokens:
            # 1. DEPENDENCY CHECK: Is the grammatical subject the player?
            subjects = [child.lower_ for child in token.children if child.dep_ == 'nsubj']
            is_self_report = any(s in PLAYER_SUBJECTS for s in subjects)
            
            if is_self_report:
                # 2. SEMANTIC MAPPING
                sent_lemmas = {t.lemma_ for t in sent}
                has_perception = bool(sent_lemmas & PERCEPTION_LEMMAS)
                has_epistemic = bool(sent_lemmas & EPISTEMIC_LEMMAS)
                
                # 3. EXPANDED SHARED CONTEXT CHECK
                subtree_words = {t.lower_ for t in token.subtree}
                is_shared = bool(subtree_words & SHARED_INDICATORS)
                
                # 4. HIGHLIGHTING FOR LLM
                start = token.idx
                end = token.idx + len(token.text)
                marked_full_text = text[:start] + "[[" + text[start:end] + "]]" + text[end:]

                findings.append({
                    'learning_verb': token.lemma_,
                    'has_perception': has_perception,
                    'has_epistemic': has_epistemic,
                    'is_shared_learning': is_shared,
                    'excerpt_sentence': sent.text,
                    'full_review_highlighted': marked_full_text
                })
                
    return findings

# --- MAIN EXECUTION ---
print("Starting analysis pipeline (Hybrid Local/Remote Mode)...")

# Ensure data directory exists if we are going to download
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)

results = []

for game_name, meta in GAME_CORPUS.items():
    print(f"Processing {game_name} ({meta['type']})...")
    
    # 1. Get Data (Local or Download)
    raw_reviews = get_game_data(meta['id'])
    print(f"  - Working with {len(raw_reviews)} reviews.")
    
    # 2. Analyze
    game_learning_events = 0
    for review_text in raw_reviews:
        if not review_text or len(review_text) < 10:
            continue
            
        events = analyze_learning_claims(review_text)
        for event in events:
            event['game'] = game_name
            event['type'] = meta['type']
            results.append(event)
            game_learning_events += 1
            
    print(f"  - Found {game_learning_events} learning claims.")

# --- OUTPUT ---
df = pd.DataFrame(results)

if not df.empty:
    print("\n--- RESULTS SUMMARY (PRE-LLM) ---")
    
    print("\n1. Shared Learning Frequency (Expanded Definition):")
    shared = df.groupby('type')['is_shared_learning'].mean() * 100
    print(shared.round(2))

    output_filename = 'steam_learning_corpus_full_context.csv'
    df.to_csv(output_filename, index=False)
    print(f"\nFull corpus with highlighted context saved to '{output_filename}'")
else:
    print("No learning claims found.")