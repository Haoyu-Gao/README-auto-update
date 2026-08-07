"""README parsing utilities.

Splits a README into paragraph-level sections (blank-line separated), maps
changed line positions to section indices (1-based), and builds the header
tree used for hierarchical recall.

Ported verbatim from the original evaluation code
(~/eval/task2_hierarchical_eval/readme_parsing.py); the parsing logic is
unchanged so that all paper numbers remain reproducible.
"""

from bs4 import BeautifulSoup
from markdown import markdown


class Node:
    def __init__(self, title, level):
        self.title = title
        self.level = level
        self.content = []
        self.children = []
        self.interval = [None, None]
        self.parent = None

    def to_dict(self):
        return {
            "title": self.title,
            "level": self.level,
            "content": self.content,
            "interval": self.interval,
            "children": [child.to_dict() for child in self.children],
        }


def build_tree_with_paragraph_mapping(paragraph_texts):
    header_info = normalize_paragraphs(paragraph_texts)

    root = Node("Document Root", 0)
    root.interval = [1, len(paragraph_texts)]

    stack = [root]
    header_nodes = []

    for idx, header in enumerate(header_info):
        if header is None:
            continue
        title, level = header
        node = Node(title=title, level=level)
        node.interval[0] = idx + 1  # 1-based index

        while stack and stack[-1].level >= level:
            popped = stack.pop()
            popped.interval[1] = node.interval[0] - 1  # close previous interval

        stack[-1].children.append(node)
        node.parent = stack[-1]
        stack.append(node)
        header_nodes.append(node)

    while stack:
        popped = stack.pop()
        popped.interval[1] = len(paragraph_texts)

    assign_content_idx(root)
    return root


def assign_content_idx(node):
    if not node.children:
        return

    min_start = min(child.interval[0] for child in node.children if child.interval[0] is not None)
    cur_start = node.interval[0]
    node.content = range(cur_start, min_start)

    for child in node.children:
        assign_content_idx(child)


def split_readme_sections_vanilla(data_entry):
    """Split `readme_base` into blank-line separated sections.

    Returns (sections, changed_sections_idx) where changed_sections_idx are the
    1-based indices of sections containing any of `line_positions`.
    """
    md_text = data_entry["readme_base"]
    line_positions = data_entry["line_positions"]
    sections = []
    section_lines = []
    current_section = []
    current_section_lines = []
    for line_pos, line in enumerate(md_text.split("\n")):
        line_pos += 1
        if line.strip():
            current_section.append(line)
            current_section_lines.append(line_pos)
        else:
            sections.append("\n".join(current_section))
            current_section_lines.append(line_pos)
            section_lines.append(current_section_lines)
            current_section = []
            current_section_lines = []
    if current_section:
        sections.append("\n".join(current_section))
        current_section_lines.append(max(current_section_lines) + 1)
        section_lines.append(current_section_lines)
    changed_sections_idx = []
    if len(line_positions) == 1 and line_positions[0] == 0:
        changed_sections_idx = [0]
    for line in line_positions:
        for section_idx, section_line in enumerate(section_lines):
            if line in section_line:
                changed_sections_idx.append(section_idx)
                break
    changed_sections_idx = sorted(set(i + 1 for i in changed_sections_idx))
    return sections, changed_sections_idx


def normalize_paragraphs(paragraph_texts):
    normalized = []
    for para in paragraph_texts:
        html = markdown(para.strip())
        soup = BeautifulSoup(html, "html.parser")
        title = ""
        for tag in soup.find_all(["h1", "h2", "h3", "h4"]):
            title = tag.get_text().strip().rstrip("#").strip()
            normalized.append((title, int(tag.name[1])))
        if title == "":
            normalized.append(None)

    return normalized
