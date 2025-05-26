"""
Date: 2025-05-24 17:39:20

Description: Gets common nouns for the roles

Input files:
- data/clean/atus_roles.txt: Cleaned roles from ATUS data
- data/clean/onet_roles.txt: Cleaned roles from ONET data

Output files:
- data/clean/common_nouns.json: Common nouns for the roles which is a json in format
    {
        "nouns": ["noun1", "noun2", ...]
    }
"""




import os
import json
import litellm


#
with open("../data/clean/atus_roles.txt", "r") as f:
    atus_roles = f.read().splitlines()

with open("../data/clean/onet_roles.txt", "r") as f:
    onet_roles = f.read().splitlines()

roles = atus_roles + onet_roles


def get_common_nouns(roles, model="gpt-4o-2024-08-06"):
    """Ask LLM for common nouns from list of roles"""
    roles_text = ", ".join(roles)

    response = litellm.completion(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
            {"role": "user", "content": f"""INSTRUCTIONS
Below are a list of roles engaged in social learning. List 20 nouns that these roles do. Each noun should be incredibly common, common to these roles, and uniquely associated with social learning, not other kinds of learning.

Roles: {roles_text}

RETURN
Return as JSON: {{"nouns": ["noun1", "noun2", ...]}}"""}
        ],
        temperature=0,
        seed=42
    )

    return json.loads(response.choices[0].message.content)


if __name__ == "__main__":
    fn = "../data/clean/common_nouns.json"
    if os.path.exists(fn):
        with open(fn, "r") as f:
            result = json.load(f)
        print("Common nouns:", result['nouns'])
        print("already did it")
        exit()

    else:
        result = get_common_nouns(roles)
        print("Common nouns:", result['nouns'])

        os.makedirs("../data/clean", exist_ok=True)
        with open("../data/clean/common_nouns.json", "w") as f:
            json.dump(result, f, indent=2)