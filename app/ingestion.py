"""Rule-based memory extraction.

Deliberately transparent: every slot is one regex plus one output template, so
it is easy to read, test and extend. A "slot" names a single-valued attribute
(OS, deadline, city...). A new memory with the same slot replaces the old one.
"""
import re
from dataclasses import dataclass


@dataclass
class CandidateMemory:
    text: str
    memory_type: str
    slot: str | None = None


SMALL_TALK = {
    "ok", "okay", "thanks", "thank you", "thanks a lot", "cool", "lol", "hmm",
    "got it", "sure", "yes", "no", "fine", "great", "nice", "hi", "hello",
    "hey", "bye", "ok thanks", "okay thanks", "ok thank you", "good morning",
    "good night", "awesome", "alright", "yep", "nope", "k",
}

DB_NAMES = {"postgresql": "PostgreSQL", "postgres": "PostgreSQL", "mysql": "MySQL",
            "sqlite": "SQLite", "mongodb": "MongoDB", "redis": "Redis"}
LANGS = r"python|java|javascript|typescript|rust|go|golang|kotlin|c\+\+|c#|c"
OSES = r"arch linux|arch|ubuntu|fedora|debian|linux mint|manjaro|nixos|windows(?: \d+)?|macos|linux"
EDITORS = r"vs ?code|vim|neovim|emacs|intellij|pycharm|sublime text|zed"

_CLAUSE = r"(?:actually,?\s*|now,?\s*|well,?\s*)?"
_I = r"(?:i|i'm|i am)"

# (slot, type, regex, template). Order matters: first match wins.
# group(1) is the value. Regexes run on a single sentence, case-insensitive.
RULES: list[tuple[str, str, str, str]] = [
    ("deadline", "event",
     rf"^{_CLAUSE}(?:my |the )?(?:project )?deadline\s+(?:is|was|has moved to|moved to|changed to|got moved to|is now|is pushed to|is extended to)\s+(.+)$",
     "Project deadline is {v}."),
    ("os", "fact",
     rf"^{_CLAUSE}(?:i (?:now )?(?:use|run|switched to|moved to|migrated to)|i'm (?:now )?(?:using|running)|i am (?:now )?(?:using|running)|my (?:os|operating system) is)\s+(?:the\s+)?({OSES})(?:\s+(?:as my|on my|for).*)?$",
     "User uses {v}."),
    ("database", "preference",
     rf"^{_CLAUSE}(?:i (?:now )?(?:prefer|use|like|switched to)|my (?:preferred|favou?rite) database is)\s+({'|'.join(DB_NAMES)})(?:\s+(?:over|instead of|rather than|for)\b.*)?$",
     "User prefers {v}."),
    ("programming_language", "preference",
     rf"^{_CLAUSE}(?:i (?:now )?(?:prefer|use|love|code in|write)|my (?:favou?rite|preferred) (?:programming )?language is)\s+({LANGS})(?:\s+(?:over|instead of|for|when|because)\b.*)?$",
     "User prefers {v}."),
    ("editor", "preference",
     rf"^{_CLAUSE}(?:i (?:now )?(?:use|prefer|switched to)|my (?:editor|ide) is)\s+({EDITORS})(?:\s+(?:as my|for).*)?$",
     "User uses {v} as their editor."),
    ("project", "project",
     rf"^{_CLAUSE}(?:i'm|i am) (?:currently )?(?:building|working on|developing|creating)\s+(.+)$",
     "User is working on {v}."),
    ("goal", "goal",
     r"^(?:my goal is(?: to)?|i want to|i aim to|i'm aiming to|i hope to|i plan to)\s+(.+)$",
     "User's goal is to {v}."),
    ("food_preference", "preference",
     r"^(?:my favou?rite food is|i love eating|i love)\s+(.+)$",
     "User's favourite food is {v}."),
    ("city", "fact",
     rf"^{_CLAUSE}(?:i live in|i'm living in|i am living in|i moved to|i'm based in|i am based in|i stay in)\s+(.+)$",
     "User lives in {v}."),
    ("employer", "fact",
     r"^(?:i work at|i work for|i'm working at|i am working at|i joined|i'm employed at)\s+(.+)$",
     "User works at {v}."),
    ("study", "fact",
     r"^(?:i study|i'm studying|i am studying|i'm a student of|my major is|my degree is(?: in)?)\s+(.+)$",
     "User studies {v}."),
    ("name", "fact", r"^(?:my name is|call me)\s+(.+)$", "User's name is {v}."),
    ("birthday", "fact", r"^my birthday is(?: on)?\s+(.+)$", "User's birthday is {v}."),
    ("laptop", "fact", r"^(?:my (?:laptop|computer|machine) is|i have|i own)\s+((?:a|an|the)\s+.*(?:laptop|thinkpad|macbook|desktop|pc).*)$",
     "User's laptop is {v}."),
    ("allergy", "fact", r"^i(?:'m| am) allergic to\s+(.+)$", "User is allergic to {v}."),
    ("hobby", "preference",
     r"^(?:i (?:really )?(?:enjoy|like|love)|my hobby is|my hobbies are)\s+(.+)$",
     "User enjoys {v}."),
    ("exam", "event", r"^(?:my )?(exam|interview|presentation|demo|hackathon)\s+(?:is|will be|was moved to|moved to)\s+(?:on\s+)?(.+)$",
     "Upcoming {v}."),
]

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
_NEGATION = re.compile(r"\b(?:no longer|don't|do not|stopped|never)\b", re.I)


def _canon(slot: str, value: str) -> str:
    if slot == "database":
        return DB_NAMES.get(value.casefold(), value)
    if slot == "os":
        v = value.strip()
        return {"arch": "Arch Linux"}.get(v.casefold(), v[:1].upper() + v[1:])
    if slot == "programming_language":
        v = value.strip()
        return {"c++": "C++", "c#": "C#", "golang": "Go"}.get(v.casefold(), v[:1].upper() + v[1:])
    return value.strip(" .!?,")


def _extract_sentence(sentence: str) -> CandidateMemory | None:
    s = sentence.strip(" \t")
    if not s or s.casefold().strip(" .!?,") in SMALL_TALK:
        return None
    if _NEGATION.search(s):          # "I no longer use X" is not a positive fact
        return None
    s_clean = s.rstrip(".!? ")
    for slot, kind, pattern, template in RULES:
        m = re.match(pattern, s_clean, flags=re.IGNORECASE)
        if not m:
            continue
        if slot == "exam":
            value = f"{m.group(1).lower()} is on {m.group(2).strip(' .!?,')}"
            return CandidateMemory(f"Upcoming {value}.", kind, f"{slot}:{m.group(1).lower()}")
        value = _canon(slot, m.group(1))
        if not value:
            return None
        return CandidateMemory(template.format(v=value), kind, slot)
    return None


def extract_memories(message: str) -> list[CandidateMemory]:
    """Return zero or more candidate memories found in a chat message."""
    clean = " ".join(message.strip().split())
    if not clean:
        return []
    out: list[CandidateMemory] = []
    seen: set[str] = set()
    for sentence in _SENT_SPLIT.split(clean):
        cand = _extract_sentence(sentence)
        if cand and cand.text.casefold() not in seen:
            seen.add(cand.text.casefold())
            out.append(cand)
    if out:
        return out
    # Fallback: keep clear first-person statements as slot-less facts.
    fold = clean.casefold()
    if re.match(r"^(i am|i'm|i have|i live|i study|i work|my name is|i like|i prefer)\b", fold) \
            and not _NEGATION.search(clean):
        kind = "preference" if re.match(r"^(i like|i prefer)\b", fold) else "fact"
        return [CandidateMemory(text=clean.rstrip(".") + ".", memory_type=kind)]
    return []
