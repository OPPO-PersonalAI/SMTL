# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 Search-R1 Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from https://github.com/PeterGriffinJin/Search-R1/blob/main/verl/utils/reward_score/qa_em.py


import datetime
import random
import re
import string

def normalize_answer(s):
    """
    Standardize the given string to ensure fair comparison during evaluation.
    This includes lowercasing, removing punctuation, removing articles, 
    and stripping extra whitespace.
    """
    def remove_articles(text):
        # Remove common English articles (a, an, the)
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        # Collapse multiple spaces into a single space and strip leading/trailing spaces
        return " ".join(text.split())

    def remove_punc(text):
        # Remove all punctuation characters defined in the string module
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        # Convert all characters to lowercase
        return text.lower()

    # Apply all text normalization functions sequentially
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    """
    Calculate the Exact Match (EM) score.
    Returns 1 if the normalized prediction exactly matches any of the 
    normalized golden answers, otherwise returns 0.
    """
    # Ensure golden_answers is a list for uniform processing
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
        
    normalized_prediction = normalize_answer(prediction)
    score = 0
    
    # Check for an exact match against each ground truth answer
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
            
    return score


def subem_check(prediction, golden_answers):
    """
    Calculate the Substring Exact Match (SubEM) score.
    Returns 1 if any of the normalized golden answers is a substring of 
    the normalized prediction, otherwise returns 0.
    Useful for evaluating generative models that may output conversational filler.
    """
    # Ensure golden_answers is a list for uniform processing
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
        
    normalized_prediction = normalize_answer(prediction)
    score = 0
    
    # Check if any ground truth answer is contained within the prediction
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
            
    return score