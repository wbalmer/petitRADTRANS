===============
AI usage policy
===============

By contributing to pRT as a co-author you agree to comply with our AI usage policy. Please indicate this agreement by signing your name below and adding your agreement through a merge request on Gitlab.

Preamble
________

AI tools such as LLM-powered inline code completion or coding agents can lead to increased coding efficiency, the implementation of numerically faster or more accurate algorithms, better code structure, etc. However, this efficiency boost can only be accepted at the expense of additional homework that you, as a developer, have to do. This homework needs to be done to maintain the quality and rigor of pRT's implementation at its long-established level (and improve on it!). If you make changes to the the code, even if checked by others through merge requests, you will be seen as the default contact and main responsible person for that part of the code, going forward. Responsibility cannot be brushed off on AI. The main purpose of this policy is that we remain in control of the code base, have experts available for all parts of the code, and learn cool and new numerical techniques and coding approaches in the process.

*Tell me and I will forget. Show me and I may remember. Let me do it and I will understand.*  -- Unknown origin

Co-developer obligations
________________________

- Before you look into the finer details of code suggested by AI, you have to make sure that you understand and agree with explicit and implicit (!) assumptions made by AI to solve your coding task.
- Then, for the AI generated code, you have to carefully review and understand what it does *line-by-line*.
- You have to read the relevant parts of the documentation for libraries, packages and functions that the AI tool introduced to understand how and why they are used, and what they do.
- If new coding concepts are introduced by the AI tool, you have to make yourself familiar with how they work, ideally by coding up a small toy example *without using AI*.
- You have to test every part of the code that you introduce: does it exhibit the intended behaviour? Does it reproduce known limiting cases and break where it is supposed to break (throwing errors, ideally)? While at it: add unit tests and make sure that your added code does not make existing tests fail.
- You have to be able to answer this question with yes: "Can I debug the code I am adding without the use of AI?" Simple example: if you removed one line of code randomly, would you be able to tell it was missing and could you write it (potentially with the help of the relevant documentation)?
- If changes in the code no longer feel like yours or you feel otherwise uneasy about them, it probably means that you have not done your homework enough yet. Do it :) !
- You have to make sure that the code changes largely follow your and pRT's internal style and only deviate if there is good reason for doing so. Again, this ensures maintainability of the code base by all co-developers.
- AI coding tools seem to be good at writing comments. But it is your obligation to make these comments yours: is there any jargon in the comments that you or other developers likely do not know? Do they sound too generic? Is there any other relevant information that you, as a pRT developer would like to add?
- You have to limit the scope of your AI usage: code maintainability relies on co-developers knowing where things are done in the code, and how. Do not unnecessarily change or refactor large parts of the code. Be as surgically precise as possible in your changes and AI coding requests.
- For now we work with this two person rule: merge requests will be completed in a one-on-one session with the review asignee and the code author, so meet either in person or virtually. This also means that you should merge often and make incremental updates to the code, whenever possible / reasonable. If a change is really tiny and straightforward, an independent (asynchronous) review by the asignee is fine, but this should be the exception and needs to be discussed with the asignee first. If you work on adding a larger new change / capability to pRT, you have to identify a team buddy to review your changes to your new branch periodically.
- If your AI generated code introduce coding concepts that are new to the merge request reviewer, this will lead to the review asignee having to do the homework, too.
- If a whole new function / part of the code was largely drafted by AI, add a corresponding comment at the top of that function and comment on it in the merge request. It is totally acceptable to use such AI-generated code as long as it is declared and the homework is done. Small in-line code completions to the code you are writing yourself or small bug fixes you worked out in interaction with AI do not need to be declared (but you need to do your homework).

Policy agreement
________________

I have read and agree with pRT's AI usage policy for co-developers:

- Paul Mollière
- David Hägele



