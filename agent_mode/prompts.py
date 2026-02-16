agent_update_classification_prompt =  """
Task description: Below is all the information about a pull request.  
Pull request description describe the purpose of the pull request. Changed files are the files that are changed in the pull request.
The README file is base README before any commits are made in this pull request.

Your goal: Decide whether README update is needed given the pull request information.

Key rules:
1. Read the README content first.  
2. Answer **"Yes"** if the pull request introduces changes that would affect information present in the README or require addition of new content at the same level of detail.
3. Please answer with "Yes" or "No". If no clear evidence, answer "No".

[begin_README]
{README_content}
[end_README]
[begin_PR_description]
{PR_description}
[end_PR_description]
[begin_commit_message]
{commit_message}
[end_commit_message]
[begin_changed_files]
{changed_file_names}
[end_changed_files]

Answer:
"""


agent_sufficient_prompt = """
        Task: Below is the information within a pull request. The README file is the base README before any commits are made, and needs update after pull request changes.
        
        The raw README file contents are divided into sections, each with a unique ID. Pull request description is the description of the changes made in the pull request.
        The context block can be empty or contain additional information such as the commit messages and changed file patches.
        Based on the provided information, determine whether the pull request description and additional context block provide **sufficient information** to identify which sections in the README should be updated. You should answer with a single word: YES or NO.


        One section needs update means content within that section should be modified, added, or removed. Or new information needs to be inserted after that section.
        One rule for the determining the section ids that needs to be updated is that the pull request descriptions should indicate same granularity of information already present in the README.
        For instance, adding a new function normally does **not** require updating a generic “Overview” section unless the change alters the overall paradigm. 

        [begin_README]
        {README_content}
        [end_README]

        [begin_PR_description]
        {PR_description}
        [end_PR_description]

        [begin_context]
        {context_block}
        [end_context]

        Answer (YES or NO):
        """


agentic_retrieval_prompt = """
        Task: Below is the information within a pull request. The README file is the base README before any commits are made, and it needs to be updated after the pull request changes.

        The pull request description outlines the purpose of the pull request. Commit information is ordered chronologically, and each commit includes a list of files changed in that commit.

        The ultimate goal is to determine which sections in the README should be updated based on the changes introduced in the pull request.

        However, the pull request description alone is not sufficient to make this determination.

        Your task is to identify the top 1 most relevant commit and the top 1 most relevant changed file within that commit that would be most helpful for deciding which sections in the README need to be updated.

        Return a JSON object in the following format:
        {{
        "file_to_check": ["path/to/file1"]
        }}

        One section needs update means content within that section should be modified, added, or removed. Or new information needs to be inserted after that section.
        One rule for the determining the section ids that needs to be updated is that the pull request descriptions should indicate same granularity of information already present in the README.
        For instance, adding a new function normally does **not** require updating a generic “Overview” section unless the change alters the overall paradigm. 

        [begin_README]
        {README_content}
        [end_README]

        [begin_PR_description]
        {PR_description}
        [end_PR_description]

        [begin_commit_info]
        {formatted_commits_data}
        [end_commit_info]

        Your output:
        """

agentic_localise_justify_prompt = """
        Task: Below is the information within a pull request. The README file is the base README before any commits are made, and needs update after pull request changes.
        
        README contents are divided into sections, each with a unique ID. Pull request descriptions is the description of the changes made in the pull request.
        In the context block, there could be additional information such as the commit messages and changed file patches, or it could be empty.

        Based on the provided information, determine which sections in the README should be updated. Return in a json format with the following keys (the returned section_ids should contains top 5 most likely section IDs, sorted by likelihood):
        {{
        "section_ids": [1, 3, 5, 6, 7],
        "justification": reasoning for why these sections should be updated given the PR description
        }}

        One section needs update means content within that section should be modified, added, or removed. Or new information needs to be inserted after that section.
        One rule for the determining the section ids that needs to be updated is that the pull request descriptions should indicate same granularity of information already present in the README.
        For instance, adding a new function normally does **not** require updating a generic “Overview” section unless the change alters the overall paradigm. 

        [begin_README]
        {README_content}
        [end_README]

        [begin_PR_description]
        {PR_description}
        [end_PR_description]

        [begin_context]
        {context_block}
        [end_context]

        Your output:
        """



justification_prompt = """
You are given:
1. A README file divided into numbered sections.
2. A list of section IDs predicted to require updates in a pull request.
3. A justification explaining why those sections were predicted.

Task:
Determine whether the justification is appropriate for the location pointed out in the corresponding README sections.
You need to read the predicted sections in the READMEs, and carefully check whether the justification provides clear and specific reasons for updating those sections, supported by evidence from the README content.

Decision rule:
Return **"Yes"** only when **all** of the following are true:

• At least part of the justification is clearly relevant to one or more of the predicted section IDs.  
• The reasoning matches the level of detail (granularity) present in the corresponding README content.  
  - For example, if the justification is about an OS version update, the referenced section must already contain an OS version to be updated.
  - Additionally, the purpose of the pull request should be clear enough, rather than vague or ambiguous guesses linking to the README.
• The README section actually covers the concept that the justification claims needs changing, not just a higher-level topic.  
  - For instance, adding a new function normally does **not** require updating a generic “Overview” section unless the change alters the overall paradigm.  
  - Likewise, if a predicted section only contains a URL to external documents, the justification must show that the URL itself (not merely the linked content) needs to change.

If **any** of these conditions are not satisfied, answer **"No"**. 
Additionally, if the justification does not align with the PR intension derived from the PR description or commit messages or too vaguely argued (e.g., section A may also be impacted), answer **"No"**.
If you think the PR does not need any README update, answer **"No"**.

Input:
[BEGIN_README]
{formatted_readme}
[END_README]

[BEGIN_PREDICTED_SECTION_IDS]
{predicted_section_ids}
[END_PREDICTED_SECTION_IDS]

[BEGIN_JUSTIFICATION]
{justification}
[END_JUSTIFICATION]

[BEGIN_PR_DESCRIPTION]
{pr_description}
[END_PR_DESCRIPTION]

[BEGIN_COMMIT_MESSAGES]
{commit_message}
[END_COMMIT_MESSAGES]

"""

agent_justification_prompt = """
You are given:
1. A README file divided into numbered sections.
2. A list of section IDs predicted to require updates in a pull request.
3. A justification explaining why those sections were predicted.

Task:
Determine whether the justification is appropriate for the location pointed out in the corresponding README sections.
You need to read the predicted sections in the READMEs, and carefully check whether the justification provides clear and specific reasons for updating those sections, supported by evidence from the README content.

Decision rule:
Return **"Yes"** only when **all** of the following are true:

• At least part of the justification is clearly relevant to one or more of the predicted section IDs.  
• The reasoning matches the level of detail (granularity) present in the corresponding README content.  
  - For example, if the justification is about an OS version update, the referenced section must already contain an OS version to be updated.
  - Additionally, the purpose of the pull request should be clear enough, rather than vague or ambiguous guesses linking to the README.
• The README section actually covers the concept that the justification claims needs changing, not just a higher-level topic.  
  - For instance, adding a new function normally does **not** require updating a generic “Overview” section unless the change alters the overall paradigm.  
  - Likewise, if a predicted section only contains a URL to external documents, the justification must show that the URL itself (not merely the linked content) needs to change.

If **any** of these conditions are not satisfied, answer either "Generic" or "Hallucinate". Generic means the justification is too vague or high-level to link to specific sections. Hallucinate means the justification contains incorrect assumptions about the README content presented.

Input:
[BEGIN_README]
{formatted_readme}
[END_README]

[BEGIN_PREDICTED_SECTION_IDS]
{predicted_section_ids}
[END_PREDICTED_SECTION_IDS]

[BEGIN_JUSTIFICATION]
{justification}
[END_JUSTIFICATION]

[BEGIN_PR_DESCRIPTION]
{pr_description}
[END_PR_DESCRIPTION]

[BEGIN_COMMIT_MESSAGES]
{commit_message}
[END_COMMIT_MESSAGES]

Output Format:
You must output a single valid JSON object containing your decision. 


Example JSON Output:
{{
    "status": "Yes" or "Generic" or "Hallucinate",
    "reason": "The justification specifically addresses the OS version mentioned in Section 3."
}}
"""



agent_guard_prompt = """
You are given:
1. A README file divided into numbered sections.
2. A list of section IDs predicted to require updates.
3. A justification explaining why those sections were predicted.

Task:
Determine if the **existing** README sections strictly require an update, or if they are already abstract enough to cover the changes.
You act as a filter to reject updates that are technically true but unnecessary because the documentation was written to be generic.

Decision Rules:

Return **"Yes"** only if:
• The code change makes the current text explicitly **factually incorrect** or **misleading**.
  - Example: The text lists specific parameters/versions that are now wrong.
• The change alters the "Public Contract" (how a user interacts with the software) in a way the current text does not capture.

Return **"No"** if:
• **The "Generic Defense":** The README section describes the behavior at a high level, and that high-level description remains true despite the code change.
  - Example: The README says "The system validates input," and the PR changes the validation library. The text "The system validates input" is still true.
• **Implementation Detail:** The change is internal (refactoring, optimization) and the current README does not expose these details.
• **Hallucination:** The justification relies on details or assumptions not present in the actual README text.

Input:
[BEGIN_README]
{formatted_readme}
[END_README]

[BEGIN_PREDICTED_SECTION_IDS]
{predicted_section_ids}
[END_PREDICTED_SECTION_IDS]

[BEGIN_JUSTIFICATION]
{justification}
[END_JUSTIFICATION]

[BEGIN_PR_DESCRIPTION]
{pr_description}
[END_PR_DESCRIPTION]

[BEGIN_COMMIT_MESSAGES]
{commit_message}
[END_COMMIT_MESSAGES]

Output Format:
You must output a single valid JSON object containing your decision.

Example JSON Output:
{{
    "status": "Yes" or "No",
    "reason": "No update needed because the README generically states 'supports data encryption', which remains true even though the underlying algorithm changed from AES-128 to AES-256."
}}
"""



