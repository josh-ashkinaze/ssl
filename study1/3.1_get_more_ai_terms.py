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

kws = {}

# Anthro Score
###################################################
###################################################
# Words from
# https://aclanthology.org/2024.eacl-long.49.pdf, Appendix B.2, page 16
anthro_score = ["palm", "lms", "llama", "transformers", "language models", "language model", "gpt", "plms",
                "pretrained language models", "gpt-2", "xlnet", "large language models", "llms", "gpt-3",
                "foundation model",
                "gptneo", "gpt-j", "chatgpt", "gpt-4"]

kws["anthro_score"] = anthro_score
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

kws["liu_et_al"] = liu_et_al
kws["liu_et_al_subset"] = liu_et_al_subset




# Chatbots.org
###################################################
###################################################

# https://www.chatbots.org/synonyms/
# This is from (https://www.chatbots.org/synonyms/) on 2025-05-26 15:06:52
# Hocking et al. (DOI: 10.11124/JBIES-20-00225) also used chatbots.org synonyms
s_string = """<li><a href="/3d_human/">3D Human</a></li><li>ACE</li><li>AI Agent</li><li>Animated Avatar</li><li>Animated Interface Agent</li><li>Animated Pedagogical Agent</li><li>Animated Talking Avatar</li><li>Anthropomorphic Agent</li><li>Anthropomorphic Dialog Agent</li><li>Anthropomorphic Interface Agent</li><li>Anthropomorphic Spoken Dialog Agent</li><li>Artificial Brand Agent (ABA)</li><li>Artificial Character</li><li><a href="/artificial_conversational_entity/">Artificial Conversational Entity</a></li><li>Artificial Human</li><li>Artificial Intelligence Chatbot</li><li>Artificial Linguistic Entity</li><li>Artificial People</li><li>Artificial Person</li><li>Artificial Talking Head</li><li>Automated Agent</li><li>Automated Attendant</li><li>Automated Chat Agent</li><li>Automated Chat Sales Associate</li><li>Automated Question Answering System</li><li>Automated Sales Agent</li><li>Automated Virtual Agent</li><li><a href="/avatar/">Avatar</a></li><li>Believable Agent</li><li>Bot</li><li><a href="/brand_agent/">Brand Agent</a></li><li>Brand Buddy</li><li>Brand Butler</li><li>Brand Character</li><li>Branded Agent</li><li>Chat Agent</li><li><a href="/chat_bot/">Chat Bot</a></li><li>Chat Robot</li><li><a href="/chatbot/">Chatbot</a></li><li><a href="/chatterbot/">Chatterbot</a></li><li><a href="/chatterbox/">Chatterbox</a></li><li>Cognitive Agent</li><li>Computerized Avatar</li><li>Computerized Virtual Person</li><li>Conversation Agent</li><li><a href="/conversational_agent/">Conversational Agent</a></li><li>Conversational Assistant</li><li><a href="/conversational_avatar/">Conversational Avatar</a></li><li>Conversational Character</li><li>Conversational Computer</li><li>Conversational Humanoid</li><li><a href="/conversational_interface/">Conversational Interface</a></li><li>Conversational Pedagogical Agent</li><li>Conversational Personal Assistant</li><li>Conversational System</li><li>Conversational User Interface</li><li>Conversive Agent</li><li>Customer Service Agent</li><li>Cyber Individual</li><li>Desktop Mate</li><li>Dialog Management System</li><li>Dialog System</li><li>Digital Animated Avatar</li><li>Digital Employee</li><li>e-Rep</li><li>ECA</li><li>Electronic Virtual Interactive Entity</li><li>Embodied  Virtual Character</li><li><a href="/embodied_agent/">Embodied Agent</a></li><li>Embodied Cognitive Agent</li><li>Embodied Communicational Agent</li><li><a href="/embodied_conversational_agent/">Embodied Conversational Agent</a></li><li>Embodied Conversational Assistant</li><li>Embodied Conversational Character</li><li><a href="/embodied_conversational_interface_agent/">Embodied Conversational Interface Agent</a></li><li>Embodied Pedagogical Agent</li><li>Embodied Pedagogical Character</li><li>Fully Embodied Conversational Avatar</li><li>Intelligent Agent</li><li>Intelligent Answering Machine</li><li>Intelligent Conversational Assistant</li><li>Intelligent Conversational Avatar</li><li>Intelligent Talking Agent</li><li><a href="/intelligent_user_interface/">Intelligent User Interface</a></li><li>Intelligent Virtual Agent</li><li>Intelligent Virtual Assistant</li><li>Intellitar</li><li>Interaction system</li><li>Interactive Agent</li><li>Interactive Conversational Assistant</li><li>Interactive Embodied Agent</li><li>Interactive Online Character</li><li>Interactive Talking Program</li><li>Interactive Virtual Agent</li><li>Interactive Voice Response</li><li>IVA</li><li>IVR</li><li>Language Bot</li><li>Lifelike Animated Character</li><li>Mascot</li><li>Natural Language System</li><li>Online Chat Agent</li><li>Pedagogic Conversational Agent</li><li>Pedagogical Agent Persona</li><li>Pedagogical Conversational Agent</li><li>Relational Agent</li><li>Screen Mate</li><li>Self Service Agent</li><li>Service Agent</li><li>Smart Virtual Assistant</li><li>Smartbot</li><li>Sociable Agent</li><li>Synthetix Agent</li><li>Talk Bot</li><li>Talking 3D Avatar</li><li>Talking Agent</li><li>Talking Avatar</li><li><a href="/talking_head/">Talking Head</a></li><li>Teachable Agent</li><li>v-Rep</li><li>VDA</li><li>Virtual Advisor</li><li><a href="/virtual_agent/">Virtual Agent</a></li><li><a href="/virtual_assistant/">Virtual Assistant</a></li><li>Virtual Avatar</li><li>Virtual Call Center Agent</li><li>Virtual Call Centre Agent</li><li>Virtual Chat Agent</li><li>Virtual Chat Expert</li><li>Virtual Coach</li><li>Virtual Consultant</li><li>Virtual Conversational Agent</li><li>Virtual Customer Agent</li><li>Virtual Customer Service Agent</li><li>Virtual Customer Support Agent</li><li>Virtual Digital Assistant</li><li>Virtual Employee</li><li>Virtual Expert</li><li>Virtual Guide</li><li>Virtual Help Desk</li><li>Virtual Host</li><li>Virtual Hostess</li><li><a href="/virtual_human/">Virtual Human</a></li><li>Virtual Human Agent</li><li>Virtual Human Avatar</li><li>Virtual Human Persona</li><li>Virtual Online Assistant</li><li>Virtual Pedagogical Agent</li><li><a href="/virtual_people/">Virtual People</a></li><li><a href="/virtual_person/">Virtual Person</a></li><li>Virtual Personal Assistant</li><li>Virtual Reality Avatar</li><li>Virtual Representative</li><li>Virtual Robot</li><li>Virtual Sales Agent</li><li>Virtual Sales Rep</li><li>Virtual Specialist</li><li>Virtual Support Agent</li><li>Virtual Teacher</li><li>Virtual Tutor</li><li>Web Agent</li></ul>
"""

def extract_synonyms_regex(html_string):
    import re
    """
    Extract synonyms from chatbot.org using regex
    """
    synonyms = []

    li_matches = re.findall(r'<li[^>]*>(.*?)</li>', html_string, re.DOTALL)

    for li_content in li_matches:
        a_match = re.search(r'<a[^>]*>(.*?)</a>', li_content)

        if a_match:
            synonyms.append(a_match.group(1).strip())
        else:
            clean_text = re.sub(r'<[^>]*>', '', li_content).strip()
            if clean_text:
                synonyms.append(clean_text)
    return synonyms


s = extract_synonyms_regex(s_string)
kws["chatbot.org"] = s


###################################################
###################################################

# write to json

import json
output_file = "../data/raw/raw_ai_terms.json"
with open(output_file, 'w') as f:
    json.dump(kws, f, indent=2)