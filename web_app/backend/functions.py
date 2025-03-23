import re
import torch
import numpy as np
import spacy
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import wikipedia

def generate_creative_content(model, tokenizer, prompt, max_length=200, num_return_sequences=3, temperature=0.6):
    """Generate creative content using the trained model with improved sentence completion."""
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)

    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        temperature=temperature,
        top_k=50,
        top_p=0.5,  
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        early_stopping=False,
        no_repeat_ngram_size=3,
        repetition_penalty=1.5
    )

    generated_texts = []
    for ids in output:
        text = tokenizer.decode(ids, skip_special_tokens=True)
        text = re.sub(r'@-@', '', text)
        text = re.sub(r'[^A-Za-z0-9\s\.\,\!\?\'\"\;\:\-]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()

        text = ensure_complete_sentences(text)

        generated_texts.append(text)

    return generated_texts

def ensure_complete_sentences(text):
    """Ensure the text ends with a complete sentence."""
    end_markers = ['.', '!', '?']
    if text and text[-1] in end_markers:
        return text
    last_end = max([text.rfind(marker) for marker in end_markers])
    if last_end != -1:
        return text[:last_end+1]
    return text

try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    print("Downloading spaCy model...")
    spacy.cli.download("en_core_web_md")
    nlp = spacy.load("en_core_web_md")

stop_words = set(stopwords.words('english'))

def evaluate_fluency(text):
    """
    Evaluate fluency by measuring:
    1. Average sentence length
    2. Lexical diversity (unique words / total words)
    3. Grammatical correctness using a heuristic approach
    """
    if not text.strip():
        return {
            'avg_sent_length': 0,
            'lexical_diversity': 0,
            'grammatical_score': 0,
            'fluency_score': 0
        }
        
    # Tokenize sentences and words
    sentences = sent_tokenize(text)
    all_words = word_tokenize(text.lower())
    words = [w for w in all_words if w.isalnum()]
    
    # Calculate average sentence length
    if len(sentences) > 0:
        avg_sent_length = len(words) / len(sentences)
    else:
        avg_sent_length = 0
        
    # Calculate lexical diversity
    if len(words) > 0:
        lexical_diversity = len(set(words)) / len(words)
    else:
        lexical_diversity = 0
        
    # Calculate a grammatical correctness score using spaCy
    doc = nlp(text)
    grammatical_score = 0
    
    # Check if we have sentences with subjects and verbs
    for sent in doc.sents:
        has_subj = any(token.dep_ in ('nsubj', 'nsubjpass') for token in sent)
        has_verb = any(token.pos_ == 'VERB' for token in sent)
        if has_subj and has_verb:
            grammatical_score += 1
    
    if len(list(doc.sents)) > 0:
        grammatical_score /= len(list(doc.sents))
        
    # Calculate overall fluency score
    fluency_score = (
        min(avg_sent_length / 15, 1) * 0.3 +  
        lexical_diversity * 0.3 +
        grammatical_score * 0.4
    )
        
    return {
        'avg_sent_length': avg_sent_length,
        'lexical_diversity': lexical_diversity,
        'grammatical_score': grammatical_score,
        'fluency_score': fluency_score
    }

def evaluate_flexibility(text):
    """
    Evaluate flexibility by analyzing:
    1. Topic diversity
    2. Semantic range
    3. Concept switching
    """
    if not text.strip():
        return {
            'topic_diversity': 0,
            'semantic_range': 0,
            'concept_transitions': 0,
            'flexibility_score': 0
        }
        
    doc = nlp(text)
    key_nouns = [token.lemma_ for token in doc if token.pos_ == 'NOUN' and token.text.lower() not in stop_words]
    topic_diversity = len(set(key_nouns)) / len(key_nouns) if key_nouns else 0
        
    sentences = list(doc.sents)
    if len(sentences) >= 2:
        sent_embeddings = np.array([sent.vector for sent in sentences])
        similarities = cosine_similarity(sent_embeddings)
        semantic_range = 1 - (np.sum(similarities) - len(sentences)) / (len(sentences) * (len(sentences) - 1))
    else:
        semantic_range = 0
        
    concept_transitions = 0
    prev_key_entities = set()
    for sent in sentences:
        sent_entities = set([token.lemma_ for token in sent 
                             if token.pos_ in ('NOUN', 'PROPN') and token.text.lower() not in stop_words])
        if prev_key_entities and (len(sent_entities.intersection(prev_key_entities)) / max(1, len(prev_key_entities)) < 0.3):
            concept_transitions += 1
        prev_key_entities = sent_entities
    concept_transitions = concept_transitions / (len(sentences) - 1) if len(sentences) > 1 else 0
        
    flexibility_score = (
        topic_diversity * 0.4 +
        semantic_range * 0.4 +
        concept_transitions * 0.2
    )
    
    return {
        'topic_diversity': topic_diversity,
        'semantic_range': semantic_range,
        'concept_transitions': concept_transitions,
        'flexibility_score': flexibility_score
    }

def evaluate_originality(text, reference_texts=None):
    """
    Alternative evaluation of originality by:
    1. Lexical novelty: proportion of unique trigrams in the text
    2. Phrase novelty: proportion of trigrams not found in the reference corpus
    3. Comparison to reference corpus via document vector similarity
    This function avoids using a rare word frequency measure.
    """
    if reference_texts is None:
        reference_texts = []
        
    if not text.strip():
        return {
            'lexical_novelty': 0,
            'phrase_novelty': 0,
            'reference_similarity': 1,  # Higher similarity means less original
            'originality_score': 0
        }
        
    # Process the text with spaCy
    doc = nlp(text)
    # Create a list of tokens (lowercase, only alphabetic)
    tokens = [token.text.lower() for token in doc if token.is_alpha]
    
    # Generate trigrams (sequences of 3 words)
    trigrams = [' '.join(tokens[i:i+3]) for i in range(len(tokens) - 2)]
    if not trigrams:
        lexical_novelty = 0
    else:
        unique_trigrams = set(trigrams)
        # Lexical novelty: how many trigrams are unique within the text
        lexical_novelty = len(unique_trigrams) / len(trigrams)
    
    # Compute phrase novelty relative to a reference corpus
    reference_trigrams = []
    for ref_text in reference_texts:
        ref_doc = nlp(ref_text)
        ref_tokens = [token.text.lower() for token in ref_doc if token.is_alpha]
        reference_trigrams.extend([' '.join(ref_tokens[i:i+3]) for i in range(len(ref_tokens) - 2)])
    # Consider a trigram novel if it doesn't appear in the reference corpus
    phrase_novelty = (sum(1 for tg in trigrams if tg not in reference_trigrams) / len(trigrams)) if trigrams else 0
        
    # Compute similarity of the generated text to the reference corpus using document vectors
    doc_vector = doc.vector
    reference_similarities = []
    for ref_text in reference_texts:
        ref_doc = nlp(ref_text)
        similarity = cosine_similarity(
            doc_vector.reshape(1, -1), 
            ref_doc.vector.reshape(1, -1)
        )[0][0]
        reference_similarities.append(similarity)
    reference_similarity = max(reference_similarities) if reference_similarities else 0
    
    # Combine the metrics into an overall originality score.
    # Here we weight:
    #   0.4 * lexical novelty + 0.3 * phrase novelty + 0.3 * (1 - reference similarity)
    originality_score = (
        lexical_novelty * 0.4 +
        phrase_novelty * 0.3 +
        (1 - reference_similarity) * 0.3
    )
    
    return {
        'lexical_novelty': lexical_novelty,
        'phrase_novelty': phrase_novelty,
        'reference_similarity': reference_similarity,
        'originality_score': originality_score
    }

def evaluate_elaboration(text):
    """
    Evaluate elaboration by analyzing:
    1. Detail density
    2. Descriptive richness (average adjectives per noun)
    3. Explanation depth (using both keywords and dependency labels)
    """
    if not text.strip():
        return {
            'detail_density': 0,
            'descriptive_richness': 0,
            'explanation_depth': 0,
            'elaboration_score': 0
        }
    
    doc = nlp(text)
    tokens = list(doc)
    
    # Detail Density: Ratio of adjectives, adverbs, and prepositions to total tokens.
    detail_tokens = [token for token in doc if token.pos_ in ('ADJ', 'ADV') or token.dep_ == 'prep']
    detail_density = len(detail_tokens) / len(tokens) if tokens else 0
    
    # Descriptive Richness: Average number of adjectives per noun.
    nouns = [token for token in doc if token.pos_ in ('NOUN', 'PROPN')]
    adjectives = [token for token in doc if token.pos_ == 'ADJ']
    if nouns:
        avg_adj_per_noun = len(adjectives) / len(nouns)
    else:
        avg_adj_per_noun = 0
    # Normalize against an expected maximum (e.g., 0.5 adjectives per noun)
    scaled_richness = min(avg_adj_per_noun / 0.5, 1)
    
    # Explanation Depth: Count both explanation keywords and adverbial clause modifiers.
    explanation_keywords = {'because', 'since', 'therefore', 'thus', 'consequently', 'due', 'hence'}
    keyword_count = sum(1 for token in doc if token.text.lower() in explanation_keywords)
    # Count adverbial clause modifiers that often indicate explanatory clauses.
    advcl_count = sum(1 for token in doc if token.dep_ == 'advcl')
    total_explanation = keyword_count + advcl_count
    sentences = list(doc.sents)
    explanation_depth = total_explanation / len(sentences) if sentences else 0
    # Assume that on average one explanation element per sentence is a high elaboration signal.
    scaled_explanation = min(explanation_depth, 1)
    
    # Final elaboration score using weighted sum.
    elaboration_score = (
        detail_density * 0.4 +
        scaled_richness * 0.3 +
        scaled_explanation * 0.3
    )
    
    return {
        'detail_density': detail_density,
        'descriptive_richness': scaled_richness,
        'explanation_depth': scaled_explanation,
        'elaboration_score': elaboration_score
    }

def evaluate_all_dimensions(text, reference_texts=None):
    """
    Evaluate all dimensions for a text.
    """
    fluency = evaluate_fluency(text)
    flexibility = evaluate_flexibility(text)
    originality = evaluate_originality(text, reference_texts)
    elaboration = evaluate_elaboration(text)
    
    creativity_score = (
        fluency['fluency_score'] * 0.25 +
        flexibility['flexibility_score'] * 0.25 +
        originality['originality_score'] * 0.25 +
        elaboration['elaboration_score'] * 0.25
    )
    
    return {
        'text': text,
        'fluency': fluency,
        'flexibility': flexibility,
        'originality': originality,
        'elaboration': elaboration,
        'creativity_score': creativity_score
    }

def get_reference_texts(query, max_results=3):
    """
    Get reference texts from Wikipedia based on a query
    
    Args:
        query (str): Search query
        max_results (int): Maximum number of results to return
        
    Returns:
        list: List of reference texts
    """
    try:
        # Perform a search on Wikipedia using the query
        search_results = wikipedia.search(query)
        print("Wikipedia search results:", search_results)
        
        # Retrieve the content of the top N results to form a reference corpus
        reference_texts = []
        for title in search_results[:max_results]:
            try:
                page = wikipedia.page(title)
                reference_texts.append(page.content)
                print(f"Retrieved content for page: {title}")
            except Exception as e:
                print(f"Could not retrieve page for {title}: {e}")
        
        return reference_texts
    except Exception as e:
        print(f"Error searching Wikipedia: {e}")
        return []