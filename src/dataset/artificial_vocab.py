"""
artificial_vocab.py
--------------------
The artificial lexicon used in Test 1.

All labels are invented words from Lewis Carroll's "Jabberwocky" (1872)
and the Alice universe (Alice's Adventures in Wonderland, 1865;
Through the Looking-Glass, 1871).

Carroll — a mathematician and logician at Oxford (Charles Dodgson) —
constructed these words with deliberate internal structure: they are
phonotactically valid, grammatically functional, and semantically empty.
They *feel* like they mean something without meaning anything at all.

That is precisely the experimental condition we need:
labels that carry no prior semantic grounding in any language,
so that all meaning must emerge from the consensus process itself.

As Carroll wrote in Through the Looking-Glass:
  "When I use a word, it means just what I choose it to mean —
   neither more nor less."
  — Humpty Dumpty

In this experiment, the agents are Humpty Dumpty.
They must choose what each word means — together.

Mapping to CIFAR-10 categories is stored ONLY in the encrypted
ground-truth file (experiments/exp_001_baseline/ground_truth.enc).
Agents never see this mapping. All grounding comes from tutor instruction.

References:
  Carroll, L. (1872). Jabberwocky. In Through the Looking-Glass.
  Carroll, L. (1865). Alice's Adventures in Wonderland. Macmillan.
"""


# ---------------------------------------------------------------------------
# The Carroll Lexicon — 10 labels for 10 CIFAR-10 categories
# ---------------------------------------------------------------------------
# Each entry documents the word's origin in Carroll's work and its
# grammatical role there — NOT its meaning (it has none).
# The agents must build that meaning from scratch.

CARROLL_VOCABULARY = {

    "vorpal": {
        "source": "Jabberwocky (1872)",
        "original_use": "Adjective — describes the sword: 'the vorpal blade'",
        "sound_character": "Sharp, precise, cutting",
        "note": "Entered tabletop gaming (D&D) as a blade of instant decapitation. "
                "The most famous Carroll neologism after 'chortle'.",
    },

    "slithy": {
        "source": "Jabberwocky (1872)",
        "original_use": "Adjective — 'the slithy toves'",
        "sound_character": "Smooth, sinuous, combining 'lithe' and 'slimy'",
        "note": "Humpty Dumpty explains it as a portmanteau — Carroll's own term "
                "for blended words.",
    },

    "mimsy": {
        "source": "Jabberwocky (1872)",
        "original_use": "Adjective — 'all mimsy were the borogoves'",
        "sound_character": "Small, frail, miserable — portmanteau of 'miserable' and 'flimsy'",
        "note": "Later adopted into English slang for 'prim and feeble'.",
    },

    "borogove": {
        "source": "Jabberwocky (1872)",
        "original_use": "Noun — a creature: 'the borogoves'",
        "sound_character": "Round, hollow-sounding, uncertain",
        "note": "Humpty Dumpty describes it as 'a thin shabby-looking bird with "
                "its feathers sticking out all round — something like a live mop'.",
    },

    "tulgey": {
        "source": "Jabberwocky (1872)",
        "original_use": "Adjective — 'the tulgey wood'",
        "sound_character": "Dense, dark, overgrown",
        "note": "Appears only once. No Humpty Dumpty gloss. Pure atmospheric invention.",
    },

    "frumious": {
        "source": "The Hunting of the Snark (1876) and Jabberwocky",
        "original_use": "Adjective — 'the frumious Bandersnatch'",
        "sound_character": "Aggressive, fuming — portmanteau of 'fuming' and 'furious'",
        "note": "Carroll explains in the Snark preface: if you have one mental "
                "image for fuming and furious simultaneously, that is frumious.",
    },

    "manxome": {
        "source": "Jabberwocky (1872)",
        "original_use": "Adjective — 'the manxome foe'",
        "sound_character": "Formidable, threatening, with faint Isle of Man resonance",
        "note": "No gloss provided. One of the least decoded Carroll words.",
    },

    "galumphing": {
        "source": "Jabberwocky (1872)",
        "original_use": "Verb/gerund — 'he went galumphing back'",
        "sound_character": "Heavy, triumphant, lumbering — portmanteau of 'gallop' and 'triumphant'",
        "note": "One of the few Carroll words fully adopted into standard English.",
    },

    "uffish": {
        "source": "Jabberwocky (1872)",
        "original_use": "Adjective — 'he stood in uffish thought'",
        "sound_character": "Huffy, pensive, somewhere between gruff and wistful",
        "note": "Carroll described it as 'a state of mind when the voice is "
                "gruffish, the manner roughish, and the temper huffish'.",
    },

    "jubjub": {
        "source": "Jabberwocky (1872) and The Hunting of the Snark (1876)",
        "original_use": "Noun — a bird: 'the jubjub bird'",
        "sound_character": "Repetitive, insistent, percussive",
        "note": "Described in the Snark as 'a desperate bird' that lives in "
                "perpetual passion. One of Carroll's most recurring creatures.",
    },
}

# Ordered list for indexed access
ARTIFICIAL_LABELS = list(CARROLL_VOCABULARY.keys())

# ---------------------------------------------------------------------------
# Instruction template
# ---------------------------------------------------------------------------
# The tutor uses this pattern to describe images without revealing
# the real category name. Only observable visual properties.

INSTRUCTION_TEMPLATE = (
    "This is a {label}. "
    "A {label} {property_1}. "
    "A {label} also {property_2}. "
    "A {label} is not a {contrasting_label}."
)

# ---------------------------------------------------------------------------
# Seed descriptions
# Populated during experiment setup from experiments/exp_001_baseline/seed_set.json
# These are the ONLY grounding events for the agents.
# ---------------------------------------------------------------------------

SEED_DESCRIPTIONS: dict = {}


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def get_label_metadata(label: str) -> dict:
    """Return Carroll metadata for a label — for documentation only, never shown to agents."""
    return CARROLL_VOCABULARY.get(label, {})


def all_labels() -> list[str]:
    return ARTIFICIAL_LABELS.copy()
