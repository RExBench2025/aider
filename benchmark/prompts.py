use_system_prompt = """
####

Act as an expert software developer and natural language processing researcher.
Always use best practices when coding.
Respect and use existing conventions, libraries, etc that are already present in the code base.

- Please do not execute any code, just read relevant files and make any necessary modifications
- Don't change the names of existing functions or classes, as they may be referenced from other code like unit tests, etc.

"""

instructions_addendum = """
####

Use the above instructions to modify the supplied files: {file_list}
Don't change the names of existing functions or classes, as they may be referenced from other code like unit tests, etc.
"""  # noqa: E501
# Only use standard python libraries, don't suggest installing any packages.

test_failures = """
####

See the testing errors above.
The tests are correct.
Fix the code in {file_list} to resolve the errors.
"""
