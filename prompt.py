from __future__ import annotations
import re
import textwrap
from dataclasses import dataclass


@dataclass
class PublicMeta:
    gamma: int
    len_public: int
    initial_token_ids: list[list[int]]
    pos_offset: int
    path: str


@dataclass
class PrivateMeta:
    prompt: Prompt
    gamma: int
    len_private: int
    len_public: int
    token_ids: list[int]
    pos_ids: list[int]
    mask_info: list[tuple[int, int]]
    replacements: dict[int, list[Replacement]]
    path: str


class Replacement:
    text: str
    prob: float
    token_ids: list[int]
    pos_ids: list[int]
    buffer_ids: list[int]

    def __init__(self, text: str, prob: float):
        self.text = text
        self.prob = prob
        self.token_ids = []
        self.pos_ids = []
        self.buffer_ids = []


def main():
    src = """
    part1
    <confidential>
    part2 <redacted> [NAME] redacted1 </redacted>
    </confidential>
    part3
    <confidential>
    part4 <redacted> redacted2 </redacted>
    part6
    <redacted> redacted3 </redacted>
    part7
    </confidential>
    part5
    """

    prompt = Prompt(src)

    print(prompt.get_all_ar_sampling_prompts())

    print(prompt)


counter = 0


def get_uid():
    global counter
    counter += 1
    return counter


def parse_tag(open_tag: str, close_tag: str, text: str, tag_factory: callable):
    result = []

    depth = 0
    buffer = ""
    i = 0

    while i < len(text):
        if text[i:i + len(open_tag)] == open_tag and depth == 0:
            if len(buffer) > 0:
                result.append(buffer)
                buffer = ""
            depth += 1
            i += len(open_tag)
            continue

        if text[i:i + len(close_tag)] == close_tag and depth == 1:
            result.append(tag_factory(buffer))
            i += len(close_tag)
            depth -= 1
            buffer = ""
            continue

        buffer += text[i]
        i += 1

    if len(buffer) > 0:
        result.append(buffer)

    return result


class Prompt:
    CONF_OPEN = "<confidential>"
    CONF_CLOSE = "</confidential>"

    elements: list[str | Confidential]

    def __init__(self, text: str):
        self.text = ' '.join(text.strip().split())
        self.elements = parse_tag("<confidential>", "</confidential>", self.text, Confidential)

    def __repr__(self):
        result = ""
        for element in self.elements:
            result += repr(element) + "\n"
        return result

    def num_redacted(self) -> int:
        count = 0
        for element in self.elements:
            if isinstance(element, Confidential):
                for e in element.elements:
                    if isinstance(e, Redacted):
                        count += 1
            else:
                if isinstance(element, Redacted):
                    count += 1
        return count

    # sampling prompt for autoregressive LLMs
    def get_ar_sampling_prompt(self, target: Redacted) -> str:
        result = []
        for element in self.elements:
            if isinstance(element, Confidential):
                result.append(element.get_ar_sampling_prompt(target))
            else:
                result.append(element)
        
        tag = target.tag if target.tag is not None else 'word'
        
        prompt = textwrap.dedent(f"""
            Your task is to guess the redacted [TARGET], which is {tag}, in the following sentence:\n\n
            <sentence>{re.sub(r'\s{2,}', ' ', ' '.join(result))}</sentence>\n\n
            One possible {tag} that fits in to [TARGET] is:""")
        
        return prompt

    # sampling prompt for fill-in-the-middle (FIM) LLMs
    def get_fim_sampling_prompt(self, target: Redacted) -> str:
        prefixes = []
        suffixes = []
        for element in self.elements:
            if isinstance(element, Confidential):
                prefix, suffix = element.get_fim_sampling_prompt(target)
                prefixes.append(prefix)
                suffixes.append(suffix)
            else:
                if len(suffixes) > 0:
                    suffixes.append(element)
                else:
                    prefixes.append(element)

        result = '<prefix>' + ' '.join(prefixes) + '<suffix>' + ' '.join(suffixes) + '<middle>'
        return result

    def get_all_ar_sampling_prompts(self) -> list[tuple[str, Redacted]]:
        prompts = []
        for element in self.elements:
            if isinstance(element, Confidential):
                for e in element.elements:
                    if isinstance(e, Redacted):
                        prompts.append((self.get_ar_sampling_prompt(e), e))
        return prompts

    def get_all_fim_sampling_prompts(self) -> list[tuple[str, Redacted]]:
        prompts = []
        for element in self.elements:
            if isinstance(element, Confidential):
                for e in element.elements:
                    if isinstance(e, Redacted):
                        prompts.append((self.get_fim_sampling_prompt(e), e))
        return prompts


class Confidential:
    text: str
    elements: list[str | Redacted]

    def __init__(self, text: str):
        self.text = text
        self.elements = []
        self.elements = parse_tag("<redacted>", "</redacted>", text, Redacted)

    def __repr__(self):
        result = "<confidential>" + "\n"
        for element in self.elements:
            result += "\t" + repr(element) + "\n"
        result += "</confidential>"
        return result

    def get_ar_sampling_prompt(self, target: Redacted):
        result = []
        for element in self.elements:
            if isinstance(element, Redacted):
                if element.text == target.text:
                    result.append('[TARGET]')
                else:
                    result.append('*' * len(element.text))
            else:
                result.append(element)
        return ' '.join(result)

    def get_fim_sampling_prompt(self, target: Redacted) -> tuple[list[str], list[str]]:
        prefixes = []
        suffixes = []
        flag = False
        for element in self.elements:
            if isinstance(element, Redacted):
                if element.text == target.text:
                    flag = True
                else:
                    if flag:
                        suffixes.append('*' * len(element.text))
                    else:
                        prefixes.append('*' * len(element.text))
            else:
                if flag:
                    suffixes.append(element)
                else:
                    prefixes.append(element)
        return prefixes, suffixes


class Redacted:
    uid: int
    text: str
    tag: str | None

    def __init__(self, text: str):
        self.uid = get_uid()
        self._parse(text)

    def _parse(self, text):
        # find [ ] pair.
        has_tag = False
        start, end = 0, 0
        for char in text:
            if char == '[':
                start = text.index(char)
                has_tag = True
            if char == ']' and has_tag:
                end = text.index(char)
                break

        if has_tag:
            self.tag = text[start + 1:end]
            self.text = text[end + 1:]
        else:
            self.tag = None
            self.text = text

        self.text = self.text.strip()

    def __repr__(self):

        if self.tag is not None:
            return f"<redacted>{self.text}</redacted> [{self.tag}]"

        return f"<redacted>{self.text}</redacted>"


if __name__ == "__main__":
    main()
