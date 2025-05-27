To expand the list of AI terms, we engaged in iterative brainstorming with Claude 3 Sonnet via the UI on 2025-05-26. 
We started with a small set of AI terms, and then iteratively expanded on this list by asking Claude to suggest additional terms, and filtering these for valid ones. The logic for doing this is 
that LLMs can generate terms that humans may miss. 

Here is a list of 1 and 2 grams to describe LLM-powered agents and chatbots that interact with users. All of these terms could show up in various contexts such as articles, products, everyday language, etc. 
[{terms_text}]

TASK 
Expand this list with 10 additional terms, not already in the list, and return terms as a comma separated list.

CONSTRAINTS 
- Optimize for precision and not recall: Do not include terms that could come up in contexts unrelated to LLM powered agents and chatbots that interact with users
- Include specific terms and not overly vague ones. For example, "neural network" and "machine learning" are too broad and not specific to user-facing LLMs
- Make sure each term is widely used and not esoteric or overly jargony
- Do not include terms that are supersets or subsets of existing terms because this will skew word counts. For example, we already have "chatbot" so do not add "user-facing chatbot"
- Do not be clever. Find the obvious terms that we missed, not the clever ones that are not widely used.

RETURN
A comma-separated list with each term encased in double quotes