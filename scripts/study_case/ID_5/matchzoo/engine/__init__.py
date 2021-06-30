# `engine` dependencies span across the entire project, so it's better to
# leave this __init__.py empty, and use `from scripts.study_case.ID_5.matchzoo.engine.package import
# x` or `from scripts.study_case.ID_5.matchzoo.engine import package` instead of `from scripts.study_case.ID_5.matchzoo
# import engine`.
