"""
Date: 2025-05-26 21:27:05

Description: Combines AI terms from a bunch of sources

Input files:
- None

Output files:
- ../data/raw/raw_ai_terms.json
"""


import pandas as pd
import os
import re
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.helpers import text2list, list2text
import logging
import os

logging.basicConfig(filename=f"{os.path.splitext(os.path.basename(__file__))[0]}.log", level=logging.INFO, format='%(asctime)s: %(message)s', filemode='w', datefmt='%Y-%m-%d %H:%M:%S', force=True)

kws = {}






# Base AI terms
###################################################
###################################################
kws['base'] = ["AI", "A.I", "artificial intelligence", "llm", "large language model"]
logging.info(f"Base AI terms: {kws['base']}")
logging.info(f"Number of base AI terms: {len(kws['base'])}")



# Anthro Score
###################################################
###################################################
# Words from
# https://aclanthology.org/2024.eacl-long.49.pdf, Appendix B.2, page 16
# However, we removed "palm" because this flagged a lot of false psoitives (e.g: palm beach)
anthro_score = ["palm", "lms", "llama", "transformers", "language models", "language model", "gpt", "plms",
                "pretrained language models", "gpt-2", "xlnet", "large language models", "llms", "gpt-3",
                "foundation model",
                "gptneo", "gpt-j", "chatgpt", "gpt-4"]

anthro_score.remove("palm")  # Remove palm due to false positives

kws["anthro_score"] = anthro_score
logging.info(f"Anthro Score terms: {anthro_score}")
logging.info(f"Number of Anthro Score terms: {len(anthro_score)}")

###################################################
###################################################

# Liu et al
###################################################
###################################################
# Words from
# https://arxiv.org/abs/2402.16039
# SM, page 2, social media search. Need to go to SM for this

liu_et_al = ["chatbot", "conversational agent", "conversational AI",
             "virtual agent", "AI bot", "conversational bot",
             "conversational system", "chat bot",
             "virtual assistant", "digital assistant", "AI assistant", "voice assistant",
             "virtual coach",
             "virtual human",
             "virtual companion",
             "AI girlfriend",
             "AI boyfriend", "AI friend", "AI wife", "AI husband", "social bot", "relational agent", "counseling agent",
             "AI therapist", "AI companion"]

# subset not related to SSL
liu_et_al_subset = ["chatbot", "conversational agent", "conversational AI",
                    "virtual agent", "AI bot", "conversational bot",
                    "conversational system", "chat bot",
                    "virtual assistant", "digital assistant", "AI assistant", "voice assistant"]

# kws["liu_et_al"] = liu_et_al
kws["liu_et_al_subset"] = liu_et_al_subset

logging.info(f"Liu et al terms: {liu_et_al_subset}")
logging.info(f"Number of Liu et al terms: {len(liu_et_al_subset)}")




# Hind and Noureddine
# https://core.ac.uk/download/pdf/642626597.pdf

###################################################
###################################################

kws["hind_noureddine"] = ['chatbots', 'virtual assistants', 'AI Agents', 'digital assistants', 'smart assistants']
logging.info(f"Hind and Noureddine terms: {kws['hind_noureddine']}")
logging.info(f"Number of Hind and Noureddine terms: {len(kws['hind_noureddine'])}")

###################################################
###################################################

all = kws.values()
all_terms = set()
for term_list in all:
    all_terms.update(term_list)
all_terms = sorted(all_terms)
print(all_terms)

logging.info(f"All terms (before plural/singular): {all_terms}")
logging.info(f"Number of all terms (before plurals/singular): {len(all_terms)}")


# ['A.I', 'AI', 'AI Agents', 'AI assistant', 'AI bot', 'artificial intelligence', 'chat bot', 'chatbot', 'chatbots', 'chatgpt', 'conversational AI', 'conversational agent', 'conversational bot', 'conversational system', 'digital assistant', 'digital assistants', 'foundation model', 'gpt', 'gpt-2', 'gpt-3', 'gpt-4', 'gpt-j', 'gptneo', 'language model',
# 'language models', 'large language model',
# 'large language models', 'llama', 'llm', 'llms',
# 'lms', 'palm', 'plms', 'pretrained language models',
# 'smart assistants', 'transformers',
# 'virtual agent', 'virtual assistant', 'virtual assistants', 'voice assistant', 'xlnet']


all_new = ['A.I', 'AI',
           'AI Agent', 'AI Agents',
           'AI assistant','AI assistants',
           'AI bot', 'AI bots',
           'artificial intelligence',
           'chat bot', 'chat bots',
           'chatbot', 'chatbots',
           'chatgpt',
           'conversational AI',
           'conversational agent', 'conversational agents',
           'conversational bot', 'conversational bots',
           'conversational system', 'conversational systems',
           'digital assistant', 'digital assistants',
           'foundation model', 'foundation models',
           'gpt', 'gpt-2', 'gpt-3', 'gpt-4', 'gpt-j', 'gptneo',
           'language model', 'language models',
           'large language model', 'large language models',
           'llama',
           'llm', 'llms',
           'lm', 'lms',
           'plm', 'plms',
           'pretrained language model', 'pretrained language models',
           'smart assistant',
           'smart assistants',
           'transformers',
           'virtual agent', 'virtual agents',
           'virtual assistant', 'virtual assistants',
           'voice assistant', 'voice assistants',
           'xlnet']



list2text("../data/clean/ai_terms.txt", all_new)
logging.info(f"All terms (after plural/singular): {all_new}")
logging.info(f"Number of all terms (after plurals/singular): {len(all_new)}")


# Make compounds
####################################
####################################


atus_roles = text2list("../data/clean/atus_roles.txt")
logging.info(f"ATUS roles: {atus_roles}")
logging.info(f"Number of ATUS roles: {len(atus_roles)}")

onet_roles = text2list("../data/clean/onet_roles.txt")
logging.info(f"ONET roles: {onet_roles}")
logging.info(f"Number of ONET roles: {len(onet_roles)}")

nouns = pd.read_json("../data/clean/common_nouns.json")['nouns'].tolist()
logging.info(f"Nouns: {nouns}")
logging.info(f"Number of nouns: {len(nouns)}")

role_predicates = ["[TERM]-powered [ROLE]",
                   "[TERM]-generated [ROLE]",
                   "[TERM]-driven [ROLE]",
                   "[TERM]-augmented [ROLE]",
                   "[TERM]-assisted [ROLE]",
                   "[TERM] [ROLE"]

logging.info(f"Role predicates: {role_predicates}")

noun_predicates  = ["[NOUN] from [TERM]",
                    "[NOUN] with [TERM]",
                    "[NOUN] using [TERM]",
                    "[NOUN] via [TERM]",
                    "[NOUN] through [TERM]",
                    "[TERM] [NOUN]"
                   ]
logging.info(f"Noun predicates: {noun_predicates}")

ai_compound_roles = [
    predicate.replace("[TERM]", term).replace("[ROLE]", role)
    for term in all_terms
    for role in atus_roles + onet_roles + nouns
    for predicate in role_predicates
]
logging.info(f"AI compound roles: {ai_compound_roles}")
logging.info(f"Number of AI compound roles: {len(ai_compound_roles)}")

ai_compound_nouns = [
    predicate.replace("[TERM]", term).replace("[NOUN]", noun)
    for term in all_terms
    for noun in nouns
    for predicate in noun_predicates
]
logging.info(f"AI compound nouns: {ai_compound_nouns}")
logging.info(f"Number of AI compound nouns: {len(ai_compound_nouns)}")

list2text("../data/clean/ai_compound_roles.txt", ai_compound_roles)
list2text("../data/clean/ai_compound_nouns.txt", ai_compound_nouns)
