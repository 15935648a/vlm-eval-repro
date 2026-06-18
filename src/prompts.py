"""Canonical prompts for the bug, plus wording variants to separate H1 (format prior)
from H2 (semantic threshold)."""

# The two prompts that disagree (kept verbatim, including the "consice" typo, so the
# reproduction matches what was originally observed).
YESNO_PROMPT = "Did the person fall in the video? Answer either yes or no"
DESCRIBE_PROMPT = (
    "Detect if any accident happens. If yes, give a very consice description, "
    "otherwise say everything is normal"
)

# The benchmark's own fixed question (from metadata.csv).
DATASET_QUESTION = "What happened? Describe in a single sentence with 2 to 6 words"

# H1 probe: is the model just biased to "No" for ANY yes/no question?
#   -> ask about something unambiguously present.
YESNO_CONTROL = "Is there a person in the video? Answer either yes or no"

# H2 probe: vary only the predicate. If "slip"/"lose balance" flip to Yes but "fall" stays
# No, the model is applying a semantic threshold ("got back up -> not a fall"), not a blanket
# format prior.
YESNO_VARIANTS = [
    "Did the person fall in the video? Answer either yes or no",
    "Did the person fall down in the video? Answer either yes or no",
    "Did the person slip in the video? Answer either yes or no",
    "Did the person lose their balance in the video? Answer either yes or no",
    "Did an accident happen in the video? Answer either yes or no",
    "Did the person fall and get back up in the video? Answer either yes or no",
]

# Keyword set used by find_video.py to locate the originally-observed clip.
TARGET_KEYWORDS = ["skat", "rink", "yellow shirt", "yellow", "getting back up", "fall"]

# Subject-referent probes. The contradictions cluster on clips whose describe-output says
# "child"/"baby"/"dog", yet the eval question asks about "the person". Vary ONLY the subject
# (predicate stays "fall") to test H4: is the word "person" the trigger for "No"?
#   - if someone/anyone/child -> Yes but person -> No  => H4 (referent mismatch)
#   - if all subjects still -> No (esp. on the adult "man" clip) => H1/H3 underneath
SUBJECT_PROBES = {
    "is_person_present": "Is there a person in the video? Answer either yes or no",
    "person_fall": "Did the person fall in the video? Answer either yes or no",
    "someone_fall": "Did someone fall in the video? Answer either yes or no",
    "anyone_fall": "Did anyone fall in the video? Answer either yes or no",
    "child_fall": "Did the child fall in the video? Answer either yes or no",
    "accident": "Did an accident happen in the video? Answer either yes or no",
}
