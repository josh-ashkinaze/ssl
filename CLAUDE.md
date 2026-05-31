

# CLAUDE.md

# Coding 
Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1\. [Step] → verify: [check]
2\. [Step] → verify: [check]
3\. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.


## 5. Formatting 
- When writing long scripts, I like functions broken up by high-level use like this, with this
particular formatting

	###############
	# Plotting functions
	###############
	def plot()
	
	###############
	# Wrangling functions
	###############
	def wrangle()
	def wrangle_more()
	
## 6. Docstrings
- Every python file needs docstrings that are formatted like this:
	
	Author: Joshua Ashkinaze
	Description: Here is what the script does
	
	Inputs:
		- input1_fn.filename: Short description of input1
		- input2_fn.filename: Short description of input2
		
	Outputs:
		- output_fn.filename: Short description of input1

# Jupyter Notebook Specific
- Just because it is a jupyter notebook, do NOT write messy code. Still prefer modular, function-based code 
- Let's say we have a long cell where we pre-process data: To make it clearer, write all the functions first and then apply them
- In a long cell, apply the same aesthetic separators as for regular code


# Data Analysis and Visualization

## Data Analysis
- Favored libraries: statsmodels, pingouin, pandas, scikit-learn, seaborn, matplotlib
- If helpers.py is available, use functions from helpers.py which have been vetted instead of rolling your own
- Do not roll your own implementation of standard things like OLS; use standard libraries in Python 
- I generally always like 95% CIs. Bootstrap if data is not too large, otherwise we can use delta method. 
- Use pandas to write latex tables; it handles a lot of the leg work already. Every table should have a name (which is the filename) and a descriptive caption.
- Write tables to latex files

## Data Viz
- If helpers.py is available, use functions from helpers.py which have been vetted for correctness instead of rolling your own
- Use make_aeshetic() from helpers.py to ensure all graphs are in a consistent format. It has similarly been vetted
- Do not write titles in the graph
- Make sure everything is legible
- For experiments only, when showing effects, draw a line at what zero would be
- When saving plots, always save as a PDF
- Try to keep consistent aspect ratios (I like 9x6)
- The goal is minimal, aesthetic, modern 
- Do not call `plt.tight_layout()`
- Do not add a ton of bells and whistles--I wrote make_aesthetic() for a reason, to have very minimal and non-fussy plots
- Use make_aesthetic() instead of defining your own font sizes unless told otherwise. make_aesthetic()
has different options for font sizes and font scale



# Academic Writing 

For academic writing, you will combine two different instructions. The first is general guidance for academic writing 
and th second is Elements of Style guidelines. In general, we want clear, concise, and active voice writing. Absolutely 
NEVER write jargon when a simple word will do, bullshit phrases that sound fancy but mean nothing, or passive voice 

## 1. General Academic Writing Guidelines
You are an expert academic writer. Write to maximize clarity, precision, and conciseness.
* If I have citations in the text, do not remove these citations.
* If I have figure or table references (e.g: ref) in the text, do not remove these references.
* If there are precent signs, format for latex (e.g: 80\%)
* If there are statistics in the text, do not remove these statistics
* If there are statistics in the text, rephrase to APA format
* Use appropriate latex syntax
* Use active voice instead of passive voice
* Use "we" statements instead of "this paper"
* Do not use idioms ever under any circumstance
* Do not delete or alter the main ideas
* Keep my style and tone of voice

## 2. Elements of Style 
- **Omit needless words.** Cut "the fact that," "due to the fact that," "in order to," "utilized," "at the present time." Every word must earn its place.
- **Use active voice.** "We ran the experiment" not "The experiment was run by us."
- **Be specific and concrete.** "It rained every day for a week" not "a period of unfavorable weather set in."
- **Put statements in positive form.** "He forgot" not "he did not remember." "Dishonest" not "not honest."
- **Place emphatic words at the end.** The end of a sentence is its most prominent position — put what matters most there.
- **Keep related words together.** Modifiers next to what they modify; don't split subject and verb with interposable phrases.
- **Parallel structure for parallel ideas.** "Both long and tedious" not "both a long ceremony and very tedious."

**Quick cuts checklist** — before finishing any prose:
* Cut: *the fact that, in order to, due to the fact that, utilized, at the present time, it is important to note that, there is/are*
* Every passive construction — is it justified, or can it be active?
* Every "not" — can it become a positive statement?
* Every *very, really, quite, certainly* — cut or replace with a strong word
* Every sentence ending weakly — restructure to end with the point




