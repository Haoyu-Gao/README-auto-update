C1_prompt_0 =  """
Task description: Below is the information within a pull request. The README file is the base README before any commits are made.
Given the pull request description, commit messages, and changed files, determine whether this README need to be updated? 
Pull request description describe the purpose of the pull request.
The README file is base README before any commits are made in this pull request.
Determine whether the README file will need to be updated given the pull request information.
Please answer with "Yes" or "No". If no clear evidence, answer "No".
[begin_README]
{README_content}
[end_README]
[begin_PR_description]
{PR_description}
[end_PR_description]
Answer:
"""



C1_prompt_1 =  """
Task description: Below is the information within a pull request. The README file is the base README before any commits are made.
Given the pull request description, commit messages, and changed files, determine whether this README need to be updated? 
Pull request description describe the purpose of the pull request. Commit messages describe each commit in a time order within the pull request.
The README file is base README before any commits are made in this pull request.
Determine whether the README file will need to be updated given the pull request information.
Please answer with "Yes" or "No". If no clear evidence, answer "No".
[begin_README]
{README_content}
[end_README]
[begin_PR_description]
{PR_description}
[end_PR_description]
[begin_commit_message]
{commit_message}
[end_commit_message]
Answer:
"""




C1_prompt_2 =  """
Task description: Below is the information within a pull request. The README file is the base README before any commits are made.
Given the pull request description, commit messages, and changed files, determine whether this README need to be updated? 
Pull request description describe the purpose of the pull request. Changed files are the files that are changed in the pull request.
The README file is base README before any commits are made in this pull request.
Determine whether the README file will need to be updated given the pull request information.
Please answer with "Yes" or "No". If no clear evidence, answer "No".
[begin_README]
{README_content}
[end_README]
[begin_PR_description]
{PR_description}
[end_PR_description]
[begin_changed_files]
{changed_file_names}
[end_changed_files]
Answer:
"""

C1_prompt_3 =  """
Task description: Below is the information within a pull request. The README file is the base README before any commits are made.
Given the pull request description, commit messages, and changed files, determine whether this README need to be updated? 
Pull request description describe the purpose of the pull request. Commit messages describe each commit in a time order within the pull request. Changed files are the files that are changed in the pull request.
The README file is base README before any commits are made in this pull request.
Determine whether the README file will need to be updated given the pull request information.
Please answer with "Yes" or "No". If no clear evidence, answer "No".
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


C1_prompt_4 =  """
Task description: Below is the information within a pull request. The README file is the base README before any commits are made.
Given the pull request description, commit messages, and changed files, determine whether this README need to be updated? 
Pull request description describe the purpose of the pull request. Commit messages describe each commit in a time order within the pull request. 
Changed files are the files that are changed in the pull request, including file names, and some patches compared to the base version.
The README file is base README before any commits are made in this pull request.
Determine whether the README file will need to be updated given the pull request information.
Please answer with "Yes" or "No". If no clear evidence, answer "No".
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
{changed_file_patches}
[end_changed_files]
Answer:
"""

C1_prompt_list = [C1_prompt_0, C1_prompt_1, C1_prompt_2, C1_prompt_3, C1_prompt_4, C1_prompt_4]


C2_sufficiency_C4 = """
        Task: Below is the information within a pull request. The README file is the base README before any commits are made, and needs update after pull request changes.
        
        Given only the pull request description and the original README, determine whether the pull request description provides **sufficient information** to identify which sections in the README should be updated.

        If it is sufficient, return in a json format with the following keys (return the top 5 most likely section IDs):
        {{
        "status": "ENOUGH",
        "section_ids": [1, 3, 5,6,7]
        "justification": reasoning for why these sections should be updated given the PR description
        }}
        If it is not sufficient and more information (e.g., commit messages or changed files) is needed to decide where the README needs update, return:
        {{
         "status": "NEED_MORE"
        }}

        Disable Chain of Thought.

        [begin_README]
        {README_content}
        [end_README]

        [begin_PR_description]
        {PR_description}
        [end_PR_description]

        Your output:
        """


C4_insufficient = """
        Task:  Below is the information within a pull request. The README file is the base README before any commits are made, and needs update after pull request changes.
        Given the pull request description, commit message, and changed file patch, identify which sections in the README need to be updated based on the changes introduced in the pull request.

        Commit messages on all commit histories in this pull request and one changed file patch identified  as the most relevant for deciding where the README needs update are provided below.
        
        Your task is to identify which sections in the README should be updated based on the changes in the pull request. Return in a json format with the following keys (return the top 5 most likely section IDs):

        {{
        "section_ids": [1, 3, 5, 6,7]
        "justification": reasoning for why these sections should be updated given the PR description
        }}

        Disable Chain of Thought.

        [begin_README]
        {README_content}
        [end_README]

        [begin_PR_description]
        {PR_description}
        [end_PR_description]

        [begin_commit_messages]
        {commit_messages}
        [end_commit_messages]

        [begin_changed_file_patch]
        {changed_file}
        [end_changed_file_patch]
        Your output:
        """



C5_prompt = """
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