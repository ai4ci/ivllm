# Principles

* If the user prompt looks like a bash command they have probably forgotten to escape it and you should stop and ask whether they forgot to escape it.
* If the input looks incomplete ask the user and wait for the next message.
* If uncertain about user intentions clarify with the user and do not make assumptions.
* Design artefacts are in the `design` folder.
* Assume any scripts in the `design` folder are instructional examples and must be completely rewritten for production.
* Anything in the `design/old` folder is superceded and likely will confuse you. Only refer to it if directly asked.
* Document architecture decisions and a phased roadmap (as a checklist) before implementation (`design/adr.md`, `design/scope.md`).
* **MAKE SURE YOU HAVE A CLEAN REVERT POINT BEFORE MAKING EVEN SEEMING MINOR CHANGES**
* All tests must be implemented and proven to fail before starting development of a feature.
* All tests must be proven to pass before development of a feature is complete.
* **IT IS NOT HELPFUL TO IMMEDIATELY FIX PRE-EXISTING BUGS WITHOUT CONSULTING THE USER**
* **IT IS ALWAYS HELPFUL TO DOCUMENT PRE-EXISTING BUGS IN `design/active-issues.md` (n.b. create it if it does not exist)**
* If you are asked to work on documentation, ONLY change documentation files:
    When documenting **do NOT fix bugs, refactor code, or modify any non-documentation files, even when you discover bugs.**
    Document discovered bugs instead (see above).
* When you discover a bug during any task:
    1. If the bug was INTRODUCED BY YOUR OWN PREVIOUS CHANGE → fix it immediately
    2. If the bug is PRE-EXISTING (introduced by the user or existed before) document it in `design/active-issues.md` and do NOT fix it
    3. It is helpful to document the suggested fix in `design/active-issues.md`
    4. If `design/active-issues.md` does not exist → create it
* Always distinguish between bugs you introduced (fix immediately) and pre-existing bugs (document and move on).
* If you are asked to create tests for existing code **CREATE TESTS BASED ON YOUR UNDERSTANDING OF INTENTION NOT EXISTING CODE BEHAVIOUR EVEN IF THEY FAIL**.
    1. Discuss new test failures with the user and document in `design/active-issues.md`.
    2. If `design/active-issues.md` does not exist → create it
* If you are asked to use a TODO list to manage a process then
    1. **CREATE THE TODO LIST**,
    2. **EXECUTE THE NEXT TASK ON THE LIST ONE AT A TIME**,
    3. **MARK THE TASK AS COMPLETE**,
    4. **REPEAT UNTIL NO MORE TASKS**. No skipping. No doing multiple steps at once.
* Before commiting confirm the following:
    1) **ALL AFFECTED TESTS PASS**,
    2) **NO NEW LINTING OR STATIC ANALYSIS ISSUES**,
    3) **DESIGN DOCUMENTATION IS UP TO DATE**,
    4) **PATCH VERSION NUMBER IS INCREMENTED**
* When you or the user have fixed an active issue remove it from `design/active-issues.md`
* git push is the users responsibility. **DO NOT TRY AND GIT PUSH THE PROJECT**

