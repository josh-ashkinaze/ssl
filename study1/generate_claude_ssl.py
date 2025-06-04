
import pandas as pd

# CLAUDE PROMPT: RUN USING UI ON 2025-06-03 11:19:37 USING CLAUDE SONNET 4

# Generate a list of LLM usages as a python dictionary with the name and definition. Some uses should be synthetic social learning and some should not be. Each should be a common usage.
#
# The definition of synthetic social learning is
#
# <start>
# The process by which humans learn through interaction with artificial agents where either (1) the knowledge acquired concerns the social world (norms, behaviors, attitudes, social skills) or (2) the interaction itself simulates social relationships or roles.
# </end>
#
# Make sure to randomize these different uses.
#
# The structure should be like a json with keys "name" and values "def" and "is_ssl"
#
# Return 40 such uses.


llm_usages = {
    "code_debugging": {
        "def": "Using LLMs to identify and fix bugs in software code across various programming languages",
        "is_ssl": False
    },
    "conversation_practice": {
        "def": "Engaging with LLMs to practice conversational skills and social interactions in various contexts",
        "is_ssl": True
    },
    "data_analysis": {
        "def": "Leveraging LLMs to analyze datasets, generate insights, and create visualizations from structured data",
        "is_ssl": False
    },
    "role_playing_therapy": {
        "def": "Using LLMs to simulate therapeutic conversations or practice difficult social scenarios",
        "is_ssl": True
    },
    "content_summarization": {
        "def": "Employing LLMs to condense long articles, documents, or reports into shorter, digestible summaries",
        "is_ssl": False
    },
    "language_translation": {
        "def": "Utilizing LLMs to translate text between different languages with contextual understanding",
        "is_ssl": False
    },
    "interview_preparation": {
        "def": "Practicing job interviews with LLMs that simulate hiring managers and provide feedback on responses",
        "is_ssl": True
    },
    "creative_writing": {
        "def": "Collaborating with LLMs to generate stories, poems, scripts, and other creative written content",
        "is_ssl": False
    },
    "social_etiquette_coaching": {
        "def": "Learning proper social behaviors and cultural norms through guided interactions with LLMs",
        "is_ssl": True
    },
    "research_assistance": {
        "def": "Using LLMs to help gather information, synthesize sources, and organize research findings",
        "is_ssl": False
    },
    "customer_service_training": {
        "def": "Training customer service representatives by simulating various customer interaction scenarios",
        "is_ssl": True
    },
    "email_composition": {
        "def": "Getting help from LLMs to write professional emails, letters, and other business correspondence",
        "is_ssl": False
    },
    "conflict_resolution_practice": {
        "def": "Practicing mediation and conflict resolution skills through simulated interpersonal disputes",
        "is_ssl": True
    },
    "mathematical_problem_solving": {
        "def": "Using LLMs to solve complex mathematical equations and explain problem-solving steps",
        "is_ssl": False
    },
    "dating_advice_simulation": {
        "def": "Receiving relationship advice and practicing dating conversations through LLM interactions",
        "is_ssl": True
    },
    "document_editing": {
        "def": "Employing LLMs to proofread, edit, and improve the quality of written documents",
        "is_ssl": False
    },
    "cultural_sensitivity_training": {
        "def": "Learning about different cultures and appropriate cross-cultural communication through LLM guidance",
        "is_ssl": True
    },
    "code_generation": {
        "def": "Having LLMs write software code based on natural language descriptions of desired functionality",
        "is_ssl": False
    },
    "parenting_guidance": {
        "def": "Seeking advice on child-rearing practices and family dynamics through conversational AI",
        "is_ssl": True
    },
    "legal_document_analysis": {
        "def": "Using LLMs to review and explain legal documents, contracts, and regulatory text",
        "is_ssl": False
    },
    "networking_skills_development": {
        "def": "Practicing professional networking conversations and learning relationship-building strategies",
        "is_ssl": True
    },
    "meal_planning": {
        "def": "Getting personalized meal plans and recipe suggestions based on dietary preferences and restrictions",
        "is_ssl": False
    },
    "public_speaking_coaching": {
        "def": "Practicing presentations and speeches with LLMs that provide audience simulation and feedback",
        "is_ssl": True
    },
    "technical_documentation": {
        "def": "Creating user manuals, API documentation, and technical guides with LLM assistance",
        "is_ssl": False
    },
    "workplace_communication_training": {
        "def": "Learning professional communication norms and practicing workplace social interactions",
        "is_ssl": True
    },
    "investment_analysis": {
        "def": "Analyzing financial markets, stocks, and investment opportunities using LLM insights",
        "is_ssl": False
    },
    "emotional_intelligence_coaching": {
        "def": "Developing empathy and emotional awareness through guided social scenario discussions",
        "is_ssl": True
    },
    "travel_planning": {
        "def": "Creating detailed travel itineraries and getting destination recommendations from LLMs",
        "is_ssl": False
    },
    "leadership_skill_practice": {
        "def": "Simulating team management scenarios to develop leadership and people management abilities",
        "is_ssl": True
    },
    "academic_tutoring": {
        "def": "Receiving personalized instruction and explanations across various academic subjects",
        "is_ssl": False
    },
    "social_media_strategy": {
        "def": "Learning about online community engagement and social media best practices through AI guidance",
        "is_ssl": True
    },
    "product_description_writing": {
        "def": "Creating compelling marketing copy and product descriptions for e-commerce platforms",
        "is_ssl": False
    },
    "small_talk_practice": {
        "def": "Improving casual conversation skills and learning appropriate topics for social chitchat",
        "is_ssl": True
    },
    "fitness_program_design": {
        "def": "Getting customized workout routines and exercise plans tailored to individual fitness goals",
        "is_ssl": False
    },
    "negotiation_skills_training": {
        "def": "Practicing business negotiations and learning persuasion techniques through simulated scenarios",
        "is_ssl": True
    },
    "grant_writing": {
        "def": "Crafting compelling grant proposals and funding applications with LLM assistance",
        "is_ssl": False
    },
    "team_building_facilitation": {
        "def": "Learning group dynamics and practicing skills for facilitating team collaboration and bonding",
        "is_ssl": True
    },
    "literature_analysis": {
        "def": "Analyzing literary works, identifying themes, and understanding complex textual meanings",
        "is_ssl": False
    },
    "cross_cultural_communication": {
        "def": "Learning appropriate communication styles and social norms for different cultural contexts",
        "is_ssl": True
    },
    "quiz_generation": {
        "def": "Creating educational quizzes and assessments for learning and knowledge evaluation purposes",
        "is_ssl": False
    }
}


df = pd.DataFrame.from_dict(llm_usages, orient='index').reset_index()
df = df.rename(columns={'index': 'name'})
df['idx'] = [i for i in range(len(df))]
df['annotation'] = None

df = df[['idx', 'name', 'def', 'annotation']]
# make excel with 3 sheets for each annotator
annotators = ["JA", "AP", "JW"]
with pd.ExcelWriter("../data/raw/pilot_annot.xlsx") as writer:
    for annotator in annotators:
        df.to_excel(writer, sheet_name=annotator, index=False)




