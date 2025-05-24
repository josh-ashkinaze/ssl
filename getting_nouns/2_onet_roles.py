import pandas as pd


# 2025-05-24 13:50:15
# o*net-soc_code         object
# element_id             object
# element_name           object
# scale_id               object
# data_value            float64
# n                       int64
# standard_error        float64
# lower_ci_bound        float64
# upper_ci_bound        float64
# recommend_suppress     object
# not_relevant           object
# date                   object
# domain_source          object

import pandas as pd

# Load and clean skills data
skills_df = pd.read_csv("https://www.onetcenter.org/dl_files/database/db_29_3_text/Skills.txt", sep="\t")
skills_df.columns = [x.lower().replace(" ", "_").replace("*", "_") for x in skills_df.columns]
skills_df = skills_df[skills_df["recommend_suppress"] == "N"]
print("Skills columns:", skills_df.columns.tolist())

# Load and clean titles data
titles_df = pd.read_csv("https://www.onetcenter.org/dl_files/database/db_29_3_text/Occupation%20Data.txt", sep="\t")
titles_df.columns = [x.lower().replace(" ", "_").replace("*", "_") for x in titles_df.columns]
print("Titles columns:", titles_df.columns.tolist())

# Merge datasets
merged = skills_df.merge(
    titles_df,
    left_on="o_net-soc_code",  # Updated column name
    right_on="o_net-soc_code", # Updated column name
    how="inner",
)

# Social skills dictionary
social_skills = {
   "Coordination": "Adjusting actions in relation to others' actions.",
   "Instructing": "Teaching others how to do something.",
   "Negotiation": "Bringing others together and trying to reconcile differences.",
   "Persuasion": "Persuading others to change their minds or behavior.",
   "Service Orientation": "Actively looking for ways to help people.",
   "Social Perceptiveness": "Being aware of others' reactions and understanding why they react as they do."
}


social_skills = {
   # "Coordination": "Adjusting actions in relation to others' actions.",
   "Instructing": "Teaching others how to do something.",
   # "Negotiation": "Bringing others together and trying to reconcile differences.",
   # "Persuasion": "Persuading others to change their minds or behavior.",
   # "Service Orientation": "Actively looking for ways to help people.",
   "Social Perceptiveness": "Being aware of others' reactions and understanding why they react as they do."
}


# Filter for social skills
social_df = merged[merged["element_name"].isin(social_skills.keys())].copy()
social_df['data_value'] = social_df['data_value'].astype(float)

# Check what scale types we have
print(f"Scale types: {social_df['scale_id'].unique()}")
print(f"Scale type counts:\n{social_df['scale_id'].value_counts()}")

# Calculate z-scores: standardize each skill by scale type across all occupations
social_df['z_data_value'] = social_df.groupby(['element_name', 'scale_id'])['data_value'].transform(
    lambda x: (x - x.mean()) / x.std() if x.std() != 0 else 0
)

print(f"\nSocial skills data shape: {social_df.shape}")
print(f"Unique skills found: {social_df['element_name'].unique()}")
print(f"Unique occupations: {social_df['o_net-soc_code'].nunique()}")

social_df2 = social_df.groupby(by=['title', 'o_net-soc_code', 'element_name']).agg(
    z_data_value=('z_data_value', 'mean'),
    data_value=('data_value', 'mean')
).reset_index()

def return_top_names(df,how='top', topn=10):
    """
    Return the top n names from the DataFrame.
    """

social_df2_top = social_df2.sort_values(by=['z_data_value'], ascending=False).head(30)['title'].to_list()
print(f"\nTop 10 occupations with highest z-scores:\n{social_df2_top}")

social_df2_bottom = social_df2.sort_values(by=['z_data_value'], ascending=True).head(10)['title'].to_list()
print(f"\nTop 10 occupations with lowest z-scores:\n{social_df2_bottom}")


# ['Mental Health Counselors',
# 'Marriage and Family Therapists',
# 'Psychiatrists',
# 'Social Work Teachers, Postsecondary',
# 'Coaches and Scouts',
# 'Clinical and Counseling Psychologists',
# 'Mental Health and Substance Abuse Social Workers',
# 'Healthcare Social Workers',
# 'Nursing Instructors and Teachers, Postsecondary',
# 'Agricultural Sciences Teachers, Postsecondary',
# 'Art, Drama, and Music Teachers, Postsecondary',
# 'Forestry and Conservation Science Teachers, Postsecondary',
# 'English Language and Literature Teachers, Postsecondary',
# 'Advanced Practice Psychiatric Nurses',
# 'Art Therapists',
# 'Architecture Teachers, Postsecondary',
# 'Substance Abuse and Behavioral Disorder Counselors',
# 'Clergy', 'Computer Science Teachers, Postsecondary', 'Geography Teachers, Postsecondary', 'Midwives', 'Clinical Neuropsychologists', 'Neuropsychologists', 'Educational, Guidance, and Career Counselors and Advisors', 'Hospitalists', 'School Psychologists', 'Communications Teachers, Postsecondary', 'Engineering Teachers, Postsecondary', 'Business Teachers, Postsecondary', 'Training and Development Managers']


