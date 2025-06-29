import re
from typing import List, Dict

# Basic IPA phonemes for English
IPA_PHONEMES = [
    'p', 'b', 't', 'd', 'k', 'g', 'f', 'v', 'θ', 'ð', 's', 'z', 'ʃ', 'ʒ', 'h',
    'm', 'n', 'ŋ', 'l', 'r', 'w', 'j',
    'i', 'ɪ', 'e', 'ɛ', 'æ', 'ʌ', 'ə', 'ɚ', 'ɝ', 'a', 'ɑ', 'ɔ', 'o', 'ʊ', 'u',
    'aɪ', 'aʊ', 'eɪ', 'oɪ', 'oʊ',
    'SIL', 'UNK'  # silence and unknown
]

# Create phoneme to index mapping
PHONE_TO_ID = {phone: i for i, phone in enumerate(IPA_PHONEMES)}
ID_TO_PHONE = {i: phone for phone, i in PHONE_TO_ID.items()}

def get_num_phones():
    """Return the number of phonemes in vocabulary."""
    return len(IPA_PHONEMES)

def text_to_phones(text: str) -> List[str]:
    """Convert text to phonemes (simplified placeholder implementation)."""
    # This is a simplified implementation. In practice, you'd use phonemizer
    # or espeak to get proper phoneme sequences.
    
    # Simple mapping for demonstration
    simple_mapping = {
        'hello': ['h', 'ɛ', 'l', 'oʊ'],
        'world': ['w', 'ɝ', 'l', 'd'],
        'the': ['ð', 'ə'],
        'a': ['ə'],
        'and': ['æ', 'n', 'd'],
        'is': ['ɪ', 'z'],
        'to': ['t', 'u'],
        'of': ['ʌ', 'v'],
        'in': ['ɪ', 'n'],
        'that': ['ð', 'æ', 't'],
        'have': ['h', 'æ', 'v'],
        'for': ['f', 'ɔ', 'r'],
        'not': ['n', 'ɑ', 't'],
        'with': ['w', 'ɪ', 'θ'],
        'he': ['h', 'i'],
        'as': ['æ', 'z'],
        'you': ['j', 'u'],
        'do': ['d', 'u'],
        'at': ['æ', 't'],
    }
    
    # Clean and tokenize text
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    words = text.split()
    
    phones = []
    for word in words:
        if word in simple_mapping:
            phones.extend(simple_mapping[word])
        else:
            # Fallback: use simple letter-to-sound rules
            for char in word:
                if char in 'aeiou':
                    phones.append('ə')  # schwa for vowels
                else:
                    phones.append(char if char in PHONE_TO_ID else 'UNK')
        phones.append('SIL')  # word boundary
    
    return phones

def phones_to_ids(phones: List[str]) -> List[int]:
    """Convert phoneme sequence to IDs."""
    return [PHONE_TO_ID.get(phone, PHONE_TO_ID['UNK']) for phone in phones]

def ids_to_phones(ids: List[int]) -> List[str]:
    """Convert ID sequence to phonemes."""
    return [ID_TO_PHONE.get(id, 'UNK') for id in ids] 